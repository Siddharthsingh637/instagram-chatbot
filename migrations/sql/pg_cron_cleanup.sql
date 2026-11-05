-- Supabase pg_cron Migration: Automatic Hourly Data Retention Cleanup
-- This script configures pg_cron on PostgreSQL (e.g. Supabase) to prune expired messages
-- and expired conversation summaries every hour, guaranteeing zero retention leakage.

-- 1. Enable the pg_cron extension
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- 2. Grant execution permission to postgres role if necessary
GRANT USAGE ON SCHEMA cron TO postgres;

-- 3. Unschedule previous job if it exists to ensure idempotency
DO $$
BEGIN
    PERFORM cron.unschedule('cleanup-expired-chat-retention');
EXCEPTION
    WHEN OTHERS THEN
        NULL;
END $$;

-- 4. Schedule the job to run every hour at minute 0
SELECT cron.schedule(
    'cleanup-expired-chat-retention',
    '0 * * * *',
    $$
    BEGIN;
        -- Delete messages past their expiration timestamp
        DELETE FROM messages 
        WHERE expires_at <= NOW();

        -- Delete conversation summaries whose retention duration has lapsed
        DELETE FROM conversation_summaries 
        WHERE expires_at IS NOT NULL AND expires_at <= NOW();
    COMMIT;
    $$
);
