#!/usr/bin/env python3
"""
Quick demo of the Music Embedding Service

Usage:
    export VOYAGE_API_KEY="your-api-key"
    python demo.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from music_embedding import MusicEmbeddingService, MusicTrack, MusicPlaylist

load_dotenv(Path(__file__).resolve().parent / ".env")


def main():
    # Check for API key
    if not os.getenv("VOYAGE_API_KEY"):
        print("Error: VOYAGE_API_KEY environment variable not set")
        print("Get your API key at: https://dashboard.voyageai.com/organization/api-keys")
        return
    
    # Initialize service with the smaller/cheaper model
    print("🎵 Initializing Music Embedding Service...")
    print("   Using model: voyage-3.5-lite ($0.02 per 1M tokens)")
    service = MusicEmbeddingService(model="voyage-3.5-lite")
    
    # Build a music catalog
    catalog = [
        MusicTrack(
            title="Purple Haze",
            artist="Jimi Hendrix",
            album="Are You Experienced",
            genre="Psychedelic Rock",
            year=1967,
            mood="energetic, trippy, intense",
            tempo="fast"
        ),
        MusicTrack(
            title="What's Going On",
            artist="Marvin Gaye",
            album="What's Going On",
            genre="Soul",
            year=1971,
            mood="reflective, soulful, mellow",
            tempo="slow"
        ),
        MusicTrack(
            title="Sweet Child O' Mine",
            artist="Guns N' Roses",
            album="Appetite for Destruction",
            genre="Hard Rock",
            year=1987,
            mood="romantic, energetic, nostalgic",
            tempo="moderate"
        ),
        MusicTrack(
            title="Superstition",
            artist="Stevie Wonder",
            album="Talking Book",
            genre="Funk",
            year=1972,
            mood="funky, upbeat, groovy",
            tempo="fast"
        ),
        MusicTrack(
            title="Hotel California",
            artist="Eagles",
            album="Hotel California",
            genre="Rock",
            year=1976,
            mood="mysterious, epic, haunting",
            tempo="moderate"
        ),
    ]
    
    print(f"\n📀 Loaded {len(catalog)} tracks into catalog")
    
    # Embed the catalog
    print("\n🔮 Embedding tracks with Voyage AI...")
    embeddings = service.embed_tracks(catalog)
    print(f"   ✓ Generated {len(embeddings)} embeddings")
    print(f"   ✓ Dimension: {len(embeddings[0])} (using 512-dim for efficiency)")
    
    # Semantic search demo
    print("\n🔍 Semantic Search Demo")
    print("=" * 50)
    
    search_queries = [
        "energetic guitar rock for a road trip",
        "chill soul music for relaxing",
        "70s funk to dance to",
    ]
    
    for query in search_queries:
        print(f"\nQuery: '{query}'")
        
        # Embed query
        query_vec = service.embed_query(query)
        
        # Simple similarity (dot product)
        import numpy as np
        query_norm = np.array(query_vec) / np.linalg.norm(query_vec)
        
        similarities = []
        for i, emb in enumerate(embeddings):
            emb_norm = np.array(emb) / np.linalg.norm(emb)
            sim = float(np.dot(query_norm, emb_norm))
            similarities.append((catalog[i], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("  Top match:")
        track, score = similarities[0]
        print(f"    → {track.title} by {track.artist} ({track.genre}, {track.year})")
        print(f"      Similarity: {score:.3f}")
    
    # Playlist embedding demo
    print("\n🎧 Playlist Embedding Demo")
    print("=" * 50)
    
    summer_vibes = MusicPlaylist(
        name="Summer Vibes 2024",
        description="A mix of feel-good tracks for beach days and sunsets",
        tracks=[catalog[0], catalog[3]],  # Purple Haze + Superstition
        mood="upbeat, nostalgic, fun",
        occasion="summer party"
    )
    
    print(f"\nEmbedding playlist: '{summer_vibes.name}'")
    print(f"   Description: {summer_vibes.description}")
    
    playlist_emb = service.embed_playlist(summer_vibes)
    print(f"   ✓ Playlist embedding: {len(playlist_emb)} dimensions")
    
    # Compare playlist to individual tracks
    print("\n   Matching tracks to playlist vibe:")
    
    import numpy as np
    playlist_norm = np.array(playlist_emb) / np.linalg.norm(playlist_emb)
    
    for track, emb in zip(catalog, embeddings):
        track_norm = np.array(emb) / np.linalg.norm(emb)
        sim = float(np.dot(playlist_norm, track_norm))
        match_emoji = "✓" if sim > 0.5 else "○"
        print(f"     {match_emoji} {track.title}: {sim:.3f}")
    
    print("\n✨ Demo complete!")
    print("\nNext steps:")
    print("  1. Store embeddings in a vector database (Pinecone, Weaviate, Chroma)")
    print("  2. Build a music recommendation API")
    print("  3. Create semantic search across your entire music library")


if __name__ == "__main__":
    main()
