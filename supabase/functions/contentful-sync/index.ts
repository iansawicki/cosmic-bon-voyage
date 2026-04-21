import { serve } from "https://deno.land/std/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const CONTENTFUL_SPACE = Deno.env.get("CONTENTFUL_SPACE_ID")!;
const CONTENTFUL_TOKEN = Deno.env.get("CONTENTFUL_DELIVERY_TOKEN")!;
const OPENAI_KEY = Deno.env.get("OPENAI_API_KEY")!;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE);

serve(async (req) => {

  try {

    const payload = await req.json();

    const entryId = payload.sys.id;
    const contentType = payload.sys.contentType.sys.id;
    const hasFields = !!payload.fields;

    console.log("Webhook received:", contentType, entryId);

    if (contentType === "track") {
      await syncTrack(entryId, hasFields);
    }

    if (contentType === "playlist") {
      await syncPlaylist(entryId);
    }

    if (contentType === "artist") {
      await syncArtist(entryId);
    }

    return new Response("ok");

  } catch (err) {

    console.error(err);

    return new Response("error", { status: 500 });

  }

});

async function fetchContentfulEntry(id: string) {

  const res = await fetch(
    `https://cdn.contentful.com/spaces/${CONTENTFUL_SPACE}/environments/master/entries/${id}?access_token=${CONTENTFUL_TOKEN}&include=2`
  );

  return await res.json();
}

async function syncTrack(id: string, hasFields: boolean) {

  if (!hasFields) {
    console.log("Track unpublished → deleting from Supabase");

    await supabase
      .from("tracks_ai")
      .delete()
      .eq("track_id", id);

    return;
  }

  const entry = await fetchContentfulEntry(id);

  const fields = entry.fields || {};

  if (!fields.title) {
  console.log("⚠️ No title found in Contentful response");
  }

  const title = fields.title || null;
  const duration = fields.duration || null;
  const audioKey = fields.audioKey || null;

  console.log("Sync track:", title);

  await supabase.from("tracks_ai").upsert({
    track_id: id,
    track_title: title,
    duration: duration,
    audio_key: audioKey
  });

  const { data } = await supabase
    .from("tracks_ai")
    .select("ai_tags_generated")
    .eq("track_id", id)
    .single();

  if (!data?.ai_tags_generated) {

    console.log("Auto tagging track");

    await autoTagTrack(id, title);

  }

}

async function autoTagTrack(trackId: string, title: string) {

  const prompt = `
Analyze this music track and return JSON with:

primary_genre
secondary_genres
style_tags
mood_keywords
themes
energy_level
summary

Track title: ${title}
`;

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: "Return ONLY valid JSON. Do not include explanations."
        },
        {
          role: "user",
          content: prompt
        }
      ]
    })
  });

  const json = await response.json();

  const tags = JSON.parse(json.choices[0].message.content);

  const combinedTags = [
    tags.primary_genre,
    ...(tags.secondary_genres || []),
    ...(tags.style_tags || []),
    ...(tags.mood_keywords || [])
  ].join(" ");

  const embed = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "text-embedding-3-small",
      input: combinedTags
    })
  });

  const embedJson = await embed.json();

  await supabase
    .from("tracks_ai")
    .update({
      primary_genre: tags.primary_genre,
      secondary_genres: tags.secondary_genres,
      style_tags: tags.style_tags,
      mood_keywords: tags.mood_keywords,
      themes: tags.themes,
      energy_level: tags.energy_level,
      summary: tags.summary,
      combined_tags: combinedTags,
      embedding: embedJson.data[0].embedding,
      ai_tags_generated: true
    })
    .eq("track_id", trackId);

}

async function syncPlaylist(id: string) {

  const entry = await fetchContentfulEntry(id);
  const fields = entry.fields || {};

  const title = fields.title || null;
  const description = fields.description || null;
  const tags = fields.tags || [];
  const tracks = fields.tracks || [];
  const curatedBy = fields.curatedBy || null;

  let backgroundImageUrl = null;

  const thumbnailRef = fields.thumbnail;

  if (thumbnailRef && thumbnailRef.sys && thumbnailRef.sys.id) {

    // Try includes first
    if (entry.includes && entry.includes.Asset) {
      const asset = entry.includes.Asset.find(
        (a: any) => a.sys.id === thumbnailRef.sys.id
      );

      if (asset && asset.fields && asset.fields.file && asset.fields.file.url) {
        backgroundImageUrl = `https:${asset.fields.file.url}`;
      }
    }

    // Fallback fetch
    if (!backgroundImageUrl) {
      const assetRes = await fetch(
        `https://cdn.contentful.com/spaces/${CONTENTFUL_SPACE}/environments/master/assets/${thumbnailRef.sys.id}?access_token=${CONTENTFUL_TOKEN}`
      );

      const assetData = await assetRes.json();

      if (assetData && assetData.fields && assetData.fields.file && assetData.fields.file.url) {
        backgroundImageUrl = `https:${assetData.fields.file.url}`;
      }
    }
  }

  console.log("Sync playlist:", title);

  await supabase.from("playlists_ai").upsert({
    playlist_id: id,
    playlist_title: title,
    description: description,
    tags: tags,
    curated_by: curatedBy,
    background_image_url: backgroundImageUrl
  });

  await supabase
    .from("track_playlist_map")
    .delete()
    .eq("playlist_id", id);

  const rows = tracks.map((track: any, index: number) => ({
    track_id: track.sys.id,
    playlist_id: id,
    position: index
  }));

  if (rows.length > 0) {
    await supabase
      .from("track_playlist_map")
      .insert(rows);
  }

  }

async function syncArtist(id: string) {

  const entry = await fetchContentfulEntry(id);

  const name = entry.fields.name?.["en-US"];

  console.log("Sync artist:", name);

  await supabase.from("artists_ai").upsert({
    artist_name: name
  });

}
