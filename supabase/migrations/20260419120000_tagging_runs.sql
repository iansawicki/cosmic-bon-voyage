-- Tagging provenance + link from tracks_ai to the run that produced current tags.
-- Apply in Supabase SQL editor or via supabase db push.

create table if not exists public.tagging_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  inference_provider text not null default 'vertex_gemini',
  model text not null,
  prompt_sha256 text,
  schema_version text,
  git_commit text,
  notes jsonb,
  vertex_batch_job text,
  batch_status text
);

create index if not exists idx_tagging_runs_created_at on public.tagging_runs (created_at desc);

alter table public.tracks_ai
  add column if not exists tagging_run_id uuid references public.tagging_runs (id);

create index if not exists idx_tracks_ai_tagging_run_id on public.tracks_ai (tagging_run_id);

comment on table public.tagging_runs is 'LLM tagging batch/run metadata (Gemini-first; legacy OpenAI batch can log here too).';
