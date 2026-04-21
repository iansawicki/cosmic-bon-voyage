-- =============================================================================
-- search_music_voyage — station search using Voyage vectors (1024-dim)
-- =============================================================================
-- Uses columns `voyage_ai_3p5_embed` on `playlists_ai` and `tracks_ai`.
-- Same shape as legacy `search_music` (curated playlists + ranked tracks).
--
-- HOW TO APPLY IN SUPABASE
-- ------------------------
-- 1. Dashboard: SQL Editor → paste this file → Run.
-- 2. Or CLI:   psql "$DATABASE_URL" -f supabase/sql_functions_local/search_music_voyage.sql
-- 3. Or copy into a new file under supabase/migrations/ and `supabase db push`
--    (recommended if you want version control applied automatically to linked projects).
--
-- BEFORE RUNNING: Ensure both tables have `voyage_ai_3p5_embed vector(1024)` (or cast
-- the function signature to match your actual column type).
--
-- AFTER RUNNING: Deploy Edge `search-music` (it calls rpc('search_music_voyage', ...)).
-- =============================================================================

CREATE OR REPLACE FUNCTION public.search_music_voyage(query_embedding vector(1024))
RETURNS TABLE (
  result_type text,
  result_id text,
  title text,
  similarity double precision
)
LANGUAGE sql
STABLE
SET search_path = public
AS $$
  (
    SELECT
      'curated_station'::text AS result_type,
      p.playlist_id::text AS result_id,
      p.playlist_title AS title,
      (1 - (p.voyage_ai_3p5_embed <=> query_embedding))::double precision AS similarity
    FROM playlists_ai AS p
    WHERE p.voyage_ai_3p5_embed IS NOT NULL
    ORDER BY p.voyage_ai_3p5_embed <=> query_embedding
    LIMIT 4
  )
  UNION ALL
  (
    SELECT
      'ai_station'::text AS result_type,
      ranked_tracks.track_id::text AS result_id,
      ranked_tracks.track_title AS title,
      ranked_tracks.similarity
    FROM (
      SELECT
        t.track_id,
        t.track_title,
        (1 - (t.voyage_ai_3p5_embed <=> query_embedding))::double precision AS similarity,
        (
          (t.voyage_ai_3p5_embed <=> query_embedding)
          - COALESCE((ts.quality_score * ts.confidence), 0) * 0.02
        ) AS ranking_score
      FROM tracks_ai AS t
      LEFT JOIN track_scores AS ts ON ts.track_id = t.track_id
      WHERE t.voyage_ai_3p5_embed IS NOT NULL
      ORDER BY ranking_score
      LIMIT 80
    ) AS ranked_tracks
  );
$$;

COMMENT ON FUNCTION public.search_music_voyage(vector(1024)) IS
  'NLP station search: Voyage embeddings on voyage_ai_3p5_embed (1024).';

GRANT EXECUTE ON FUNCTION public.search_music_voyage(vector(1024)) TO service_role;
GRANT EXECUTE ON FUNCTION public.search_music_voyage(vector(1024)) TO authenticated;
GRANT EXECUTE ON FUNCTION public.search_music_voyage(vector(1024)) TO anon;
