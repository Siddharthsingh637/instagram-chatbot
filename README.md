# AI-Powered Instagram DM Chatbot Backend

A production-ready FastAPI backend for an AI-powered Instagram DM chatbot that receives Instagram Messaging webhook events from Meta, stores and tracks messages, evaluates whether to speak, generates contextual replies with LLM providers (Groq primary, OpenRouter fallback, Gemini search grounding stub), and sends them back via the Meta Graph API.

---

## Core Architecture & Principles

1. **Platform Independence**: The AI decision and conversation engine is decoupled from Instagram-specific I/O, allowing other messaging platforms (such as Telegram or Discord) to be plugged in seamlessly.
2. **Two Decoupled Pipelines**:
   - **Pipeline A (UNDERSTAND)**: Ingests, stores, and tracks *every* message, updates conversation state, and performs event deduplication (`message.mid`). No LLM calls are made here.
   - **Pipeline B (SPEAK)**: Evaluates whether to reply using cheap deterministic rules first (global pause, conversation pause, human handoff, reply cooldown, no consecutive bot replies, trigger mode, relevance threshold), followed by a single structured LLM evaluation and context-aware reply generation.
3. **Owner-Controlled Two-Tier Memory**:
   - **Ephemeral Working Context**: The `messages` and `conversation_summaries` tables auto-expire past `expires_at` (`now + MESSAGE_RETENTION_HOURS`).
   - **Permanent Approved Memories**: Long-term facts (`memories` table) are extracted in background batches, filtered for sensitive data, and saved with `pending` status until approved by the owner via the Admin API (or auto-approved if specified in config). Only `approved` memories are ever injected into prompts.

---

## Directory Structure

```text
.
├── app/
│   ├── main.py                     # FastAPI application setup and lifespans
│   ├── api/
│   │   ├── health.py               # GET /health with DB connectivity check
│   │   ├── webhook.py              # GET /webhook (Meta verify) & POST /webhook (HMAC check)
│   │   ├── dev.py                  # POST /dev/simulate-message (local testing without Meta)
│   │   └── admin.py                # Admin API for conversations, memories, settings & cleanup
│   ├── core/
│   │   ├── config.py               # Pydantic Settings & environment validation
│   │   ├── security.py             # HMAC-SHA256 constant-time check & admin key verification
│   │   └── logging.py              # Structured logging with token/credential masking
│   ├── database/
│   │   ├── connection.py           # Sync SQLAlchemy 2.0 engine & session dependency
│   │   ├── models.py               # 10 UUID-keyed SQLAlchemy models
│   │   └── repositories.py         # Data access helpers & query utilities
│   ├── integrations/
│   │   └── instagram/
│   │       ├── client.py           # Instagram client interface
│   │       ├── events.py           # Meta webhook parser, echo filtering
│   │       └── sender.py           # Send API (/me/messages) with retries & backoff
│   ├── ai/
│   │   ├── providers/
│   │   │   ├── base.py             # Abstract AIProvider & JSON extractor
│   │   │   ├── groq_provider.py    # Primary high-throughput provider
│   │   │   ├── openrouter_provider.py # Fallback provider
│   │   │   └── gemini_provider.py  # Search grounding stub
│   │   ├── provider_factory.py     # Provider selection & automatic fallback client
│   │   ├── decision_engine.py      # Tiered cheap rules + structured LLM decision
│   │   ├── persona.py              # Dynamic persona loader & sensitivity overrides
│   │   ├── orchestrator.py         # Pipeline A + Pipeline B orchestrator
│   │   └── prompts/
│   │       ├── system.txt          # Core system prompt & guidelines
│   │       ├── professional.txt    # Professional persona template
│   │       ├── casual.txt          # Casual persona template
│   │       └── humorous.txt        # Humorous persona template (tones down on sensitivity)
│   ├── memory/
│   │   ├── manager.py              # Sensitive data filter & candidate memory extraction
│   │   ├── retrieval.py            # Approved & pending memory queries
│   │   ├── summarizer.py           # Expiry-synchronized conversation summaries
│   │   └── retention.py            # Hourly message & summary deletion logic
│   ├── notifications/
│   │   ├── base.py                 # Abstract OwnerNotifier interface
│   │   └── owner_notifier.py       # Default logging notifier (pluggable for Telegram/Email)
│   └── workers/
│       └── message_worker.py       # Background worker for webhook payloads
├── migrations/                     # Alembic database migrations
│   ├── sql/pg_cron_cleanup.sql    # Supabase pg_cron hourly retention job
│   └── versions/                   # Migration versions
├── tests/                          # 40 comprehensive pytest unit & integration tests
├── .env.example                    # Sample environment variables
├── .gitignore                      # Git ignore excluding .env, DBs, and virtualenvs
├── alembic.ini                     # Alembic configuration
├── pytest.ini                      # Pytest test configuration
├── requirements.txt                # Production and test Python dependencies
└── README.md                       # Documentation & setup guide
```

---

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `APP_ENV` | Yes | `development` | `development`, `production`, or `test`. |
| `DATABASE_URL` | Yes | `sqlite:///./instagram_bot.db` | PostgreSQL URL for production/Supabase, SQLite for dev/test. |
| `AI_PROVIDER` | Yes | `groq` | Primary AI provider: `groq`, `openrouter`, or `gemini`. |
| `GROQ_API_KEY` | If using Groq | `None` | API key from Groq Console. |
| `OPENROUTER_API_KEY` | Optional | `None` | API key from OpenRouter (used for fallback). |
| `GEMINI_API_KEY` | Optional | `None` | API key from Google AI Studio. |
| `META_APP_SECRET` | In Prod | `None` | Meta App Secret for HMAC-SHA256 signature verification. |
| `META_VERIFY_TOKEN` | In Prod | `None` | Secret token configured in Meta App Dashboard for webhook verification. |
| `INSTAGRAM_ACCESS_TOKEN`| In Prod | `None` | Page Access Token with `instagram_manage_messages` permission. |
| `INSTAGRAM_ACCOUNT_ID` | In Prod | `None` | Connected Instagram Business / Creator Account ID. |
| `GRAPH_API_VERSION` | No | `v21.0` | Meta Graph API version. |
| `ADMIN_API_KEY` | Yes | `None` | Secret key required in the `X-Admin-Key` header for admin & dev routes. |
| `MESSAGE_RETENTION_HOURS`| No | `6` | Hours before stored messages expire. |
| `REPLY_COOLDOWN_SECONDS` | No | `10` | Minimum seconds between consecutive bot replies in a conversation. |
| `MAX_CONTEXT_MESSAGES` | No | `15` | Maximum previous messages included in the LLM prompt. |
| `HANDOFF_KEYWORDS` | No | `human,agent,real person` | Comma-separated list of words triggering human handoff. |
| `AUTO_APPROVED_MEMORY_CATEGORIES` | No | `""` | Optional comma-separated list of categories to auto-approve (e.g. `preferences,hobbies`). |

---

## Local Setup & Running Tests

### 1. Create and Activate Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Alembic Migrations
```bash
alembic upgrade head
```

### 3. Run Test Suite
All 40 unit and integration tests execute with an in-memory SQLite database and test doubles (no external API calls):
```bash
pytest -v
```

---

## Running the Application Locally

Start the Uvicorn development server:
```bash
uvicorn app.main:app --reload --port 8000
```

Verify health check:
```bash
curl http://localhost:8000/health
```
Response:
```json
{"status": "healthy", "database": "connected", "app_env": "development"}
```

---

## Testing Locally Without Meta

When `APP_ENV=development`, you can simulate incoming Instagram messages through the development endpoint without configuring Meta webhooks:

```bash
curl -X POST http://localhost:8000/dev/simulate-message \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your_admin_secret_key_here" \
  -d '{
    "sender_id": "test_user_123",
    "text": "Hello, what services do you provide?"
  }'
```

To test human handoff:
```bash
curl -X POST http://localhost:8000/dev/simulate-message \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your_admin_secret_key_here" \
  -d '{
    "sender_id": "test_user_123",
    "text": "Can I please speak to a human?"
  }'
```

To test memory clearing:
```bash
curl -X POST http://localhost:8000/dev/simulate-message \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your_admin_secret_key_here" \
  -d '{
    "sender_id": "test_user_123",
    "text": "forget me"
  }'
```

---

## Exposing Locally for Meta Testing

Meta requires an HTTPS endpoint to verify and deliver webhooks. You can use **ngrok** or **cloudflared**:

### Using ngrok:
```bash
ngrok http 8000
```
Copy the forwarding HTTPS URL (e.g. `https://your-subdomain.ngrok-free.app`).

### Webhook URLs in Meta:
- **Callback URL**: `https://your-subdomain.ngrok-free.app/webhook`
- **Verify Token**: Must match your `META_VERIFY_TOKEN` in `.env`.

---

## Meta Developer Portal Configuration (Manual Steps)

To connect the bot to live Instagram DMs, follow these steps in Meta's Developer Portal:

1. **Meta Developer Account & App**:
   - Go to [Meta for Developers](https://developers.facebook.com/).
   - Click **My Apps** > **Create App**.
   - Choose **Business** or **Other** as the app type.
2. **Add Instagram Messaging**:
   - In the App Dashboard, click **Add Product** and select **Instagram** / **Messenger**.
3. **Connect Facebook Page & Instagram Account**:
   - Your Instagram account must be an **Instagram Professional / Creator / Business Account**.
   - Link your Instagram account to a **Facebook Page** under your Meta Business Suite.
4. **Configure Webhook Subscription**:
   - Under App Dashboard > Webhooks > Select **Instagram**.
   - Set Callback URL: `https://<YOUR_DEPLOYED_DOMAIN>/webhook`.
   - Set Verify Token: The same string configured in `META_VERIFY_TOKEN`.
   - Click **Verify and Save**. Meta will send a GET request to `/webhook` with `hub.challenge`.
   - Subscribe to the **`messages`** and **`messaging_postbacks`** fields.
5. **Generate Page Access Token**:
   - In Meta Graph API Explorer or App Dashboard, generate a Page Access Token with the following permissions:
     - `instagram_basic`
     - `instagram_manage_messages`
     - `pages_manage_metadata`
     - `pages_read_engagement`
   - Store this token in `INSTAGRAM_ACCESS_TOKEN`.
6. **Obtain App Secret**:
   - In App Dashboard > **App Settings** > **Basic**, reveal the **App Secret**.
   - Store this in `META_APP_SECRET` for HMAC-SHA256 signature verification.

---

## Supabase PostgreSQL & pg_cron Setup

1. Create a Supabase PostgreSQL database.
2. Set `DATABASE_URL=postgresql+psycopg2://postgres:<password>@db.<ref>.supabase.co:5432/postgres`.
3. Apply migrations:
   ```bash
   alembic upgrade head
   ```
4. Run the hourly cron cleanup script in the Supabase SQL Editor:
   - Copy contents of `migrations/sql/pg_cron_cleanup.sql` into the Supabase SQL editor and execute.
   - This schedules an hourly job `cleanup-expired-chat-retention` that deletes expired messages and expired summaries automatically.

---

## Deploying to Render

1. **Create a New Web Service** on [Render](https://render.com/).
2. Connect your GitHub repository.
3. Configure settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && alembic upgrade head`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
4. Set Environment Variables in Render:
   - `APP_ENV`: `production`
   - `DATABASE_URL`: Your Supabase connection string.
   - `AI_PROVIDER`: `groq`
   - `GROQ_API_KEY`: Your Groq API key.
   - `META_APP_SECRET`: Your Meta App Secret.
   - `META_VERIFY_TOKEN`: Your Meta Webhook verify token.
   - `INSTAGRAM_ACCESS_TOKEN`: Your Page Access Token.
   - `INSTAGRAM_ACCOUNT_ID`: Your Instagram Account ID.
   - `ADMIN_API_KEY`: A strong random string for administrative operations.

---

## Admin API Reference

All requests must include the header:
```http
X-Admin-Key: <ADMIN_API_KEY>
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/admin/conversations` | List recent conversations with status and handoff state |
| `GET` | `/admin/conversations/{id}/messages` | List messages for a specific conversation |
| `POST` | `/admin/conversations/{id}/handoff/resume` | Resume automated bot replies after handoff |
| `GET` | `/admin/memories/pending` | List candidate memories awaiting review |
| `GET` | `/admin/memories/{user_id}` | List all memories for a user |
| `POST` | `/admin/memories/{id}/approve` | Approve a candidate memory for prompt injection |
| `POST` | `/admin/memories/{id}/reject` | Reject a candidate memory |
| `PATCH` | `/admin/memories/{id}` | Edit memory content or category |
| `DELETE` | `/admin/memories/{id}` | Permanently delete a memory record |
| `GET` | `/admin/settings` | Retrieve current bot operational thresholds |
| `PATCH` | `/admin/settings` | Update cooldown, response frequency, relevance threshold |
| `POST` | `/admin/settings/pause` | Globally pause bot replies |
| `POST` | `/admin/settings/resume` | Globally resume bot replies |
| `POST` | `/admin/cleanup` | Trigger manual retention cleanup of expired data |
