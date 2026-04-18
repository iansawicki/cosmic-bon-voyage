# 🎵 Cosmic Bon Voyage - Music Embedding Service

A Python service for embedding music metadata (tracks, albums, artists, playlists) using the Voyage AI API.

## Quick Start

1. **Get API Key**
   ```bash
   # Sign up at https://dashboard.voyageai.com/ and create an API key
   export VOYAGE_API_KEY="your-api-key-here"
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Demo**
   ```bash
   python demo.py
   ```

## Features

- 🎼 **Track Embeddings** - Embed individual songs with rich metadata (genre, mood, tempo, year)
- 📀 **Playlist Embeddings** - Embed entire playlists for vibe-based matching
- 🔍 **Semantic Search** - Find similar music using natural language queries
- 💰 **Cost Efficient** - Uses `voyage-3.5-lite` at $0.02 per 1M tokens
- ⚡ **Fast & Scalable** - Async support, batch processing, connection pooling

## Usage

### Basic Track Embedding

```python
from music_embedding import MusicEmbeddingService, MusicTrack

service = MusicEmbeddingService()

track = MusicTrack(
    title="Bohemian Rhapsody",
    artist="Queen",
    album="A Night at the Opera",
    genre="Rock",
    year=1975,
    mood="epic, dramatic",
    tempo="moderate"
)

embedding = service.embed_track(track)
print(f"Generated {len(embedding)}-dimensional embedding")
```

### Semantic Search

```python
# Embed a natural language query
query = "upbeat 70s rock for a road trip"
query_embedding = service.embed_query(query)

# Compare to your catalog (uses cosine similarity)
similarities = service.find_similar_tracks(query_track, catalog, top_k=5)
```

### Batch Processing

```python
tracks = [track1, track2, track3, ...]
embeddings = service.embed_tracks(tracks)  # More efficient than one-by-one
```

## Models

| Model | Price (per 1M tokens) | Best For |
|-------|----------------------|----------|
| `voyage-3.5-lite` | $0.02 | Cost-sensitive, high volume |
| `voyage-3.5` | $0.06 | Better quality, balanced cost |
| `voyage-4` | Higher | Latest generation, best quality |

Default is `voyage-3.5-lite` for cost efficiency.

## Embedding Dimensions

Voyage models support multiple output dimensions (controlled by truncation):
- 256-dim: Ultra-compact, fastest retrieval
- 512-dim: Balanced (default in this service)
- 1024-dim: Higher fidelity
- 2048-dim: Maximum quality

## Architecture

```
MusicTrack/Playlist → Text Description → Voyage API → Embedding Vector
```

The service converts music metadata into rich descriptive text before embedding:
- "Song: Purple Haze | Artist: Jimi Hendrix | Genre: Psychedelic Rock | Mood: energetic, trippy"

This gives the embedding model semantic context about the music.

## Use Cases

- **Music Recommendation** - "If you like X, try Y"
- **Playlist Generation** - Build playlists from natural language descriptions
- **Semantic Search** - "Find me chill jazz from the 60s"
- **Duplicate Detection** - Identify similar tracks across sources
- **Vibe Matching** - Match songs to activities, moods, occasions

## Next Steps

- Store embeddings in a vector database (Pinecone, Weaviate, Chroma)
- Build a REST API for real-time recommendations
- Add reranking with Voyage AI rerankers
- Implement hybrid search (embeddings + metadata filters)

## API Reference

### MusicEmbeddingService

```python
MusicEmbeddingService(
    api_key=None,           # Uses VOYAGE_API_KEY env var if not provided
    model="voyage-3.5-lite",
    max_retries=3,
    timeout=30
)
```

### Methods

- `embed_track(track)` - Embed a single track
- `embed_tracks(tracks)` - Batch embed tracks
- `embed_playlist(playlist)` - Embed a playlist
- `embed_playlists(playlists)` - Batch embed playlists
- `embed_query(query)` - Embed a search query
- `find_similar_tracks(query_track, candidates, top_k=5)` - Find similar tracks

## License

Cosmic Radio Internal Use 🎩
