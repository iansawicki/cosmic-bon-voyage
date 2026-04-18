"""
Music Information Embedding Service using Voyage AI

This module provides functionality to embed music information (tracks, albums,
artists, playlists) using Voyage AI's embedding models.

Set your VOYAGE_API_KEY environment variable before using.
"""

import os
import voyageai
from typing import List, Dict, Union, Optional
from dataclasses import dataclass


@dataclass
class MusicTrack:
    """Represents a music track with metadata."""
    title: str
    artist: str
    album: str
    genre: Optional[str] = None
    year: Optional[int] = None
    lyrics_snippet: Optional[str] = None
    mood: Optional[str] = None
    tempo: Optional[str] = None  # e.g., "fast", "slow", "moderate"


@dataclass
class MusicPlaylist:
    """Represents a playlist with metadata."""
    name: str
    description: str
    tracks: List[MusicTrack]
    mood: Optional[str] = None
    occasion: Optional[str] = None  # e.g., "workout", "study", "party"


class MusicEmbeddingService:
    """
    Service for embedding music information using Voyage AI.
    
    Uses voyage-3.5-lite by default for cost efficiency,
    but can be configured to use other models.
    """
    
    DEFAULT_MODEL = "voyage-3.5-lite"
    DEFAULT_DIMENSION = 512  # Can be 256, 512, 1024, or 2048
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: Voyage API key. If None, uses VOYAGE_API_KEY env var.
            model: Model to use (voyage-3.5-lite, voyage-3.5, voyage-4, etc.)
            max_retries: Number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.client = voyageai.Client(
            api_key=self.api_key,
            max_retries=max_retries,
            timeout=timeout
        )
    
    def _track_to_text(self, track: MusicTrack) -> str:
        """Convert a MusicTrack to descriptive text for embedding."""
        parts = [
            f"Song: {track.title}",
            f"Artist: {track.artist}",
            f"Album: {track.album}",
        ]
        
        if track.genre:
            parts.append(f"Genre: {track.genre}")
        if track.year:
            parts.append(f"Year: {track.year}")
        if track.mood:
            parts.append(f"Mood: {track.mood}")
        if track.tempo:
            parts.append(f"Tempo: {track.tempo}")
        if track.lyrics_snippet:
            parts.append(f"Lyrics: {track.lyrics_snippet}")
        
        return " | ".join(parts)
    
    def _playlist_to_text(self, playlist: MusicPlaylist) -> str:
        """Convert a MusicPlaylist to descriptive text for embedding."""
        parts = [
            f"Playlist: {playlist.name}",
            f"Description: {playlist.description}",
        ]
        
        if playlist.mood:
            parts.append(f"Mood: {playlist.mood}")
        if playlist.occasion:
            parts.append(f"Occasion: {playlist.occasion}")
        
        # Include track summaries
        if playlist.tracks:
            artists = list(set([t.artist for t in playlist.tracks]))[:5]
            genres = list(set([t.genre for t in playlist.tracks if t.genre]))[:3]
            
            parts.append(f"Artists: {', '.join(artists)}")
            if genres:
                parts.append(f"Genres: {', '.join(genres)}")
            parts.append(f"Track count: {len(playlist.tracks)}")
        
        return " | ".join(parts)
    
    def embed_tracks(
        self,
        tracks: List[MusicTrack],
        input_type: str = "document",
        truncation: bool = True
    ) -> List[List[float]]:
        """
        Embed a list of music tracks.
        
        Args:
            tracks: List of MusicTrack objects
            input_type: "document" or "query" (affects how model treats input)
            truncation: Whether to truncate inputs exceeding context limit
            
        Returns:
            List of embedding vectors (one per track)
        """
        texts = [self._track_to_text(track) for track in tracks]
        
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
            truncation=truncation
        )
        
        return result.embeddings
    
    def embed_track(self, track: MusicTrack) -> List[float]:
        """Embed a single music track."""
        return self.embed_tracks([track])[0]
    
    def embed_playlists(
        self,
        playlists: List[MusicPlaylist],
        input_type: str = "document",
        truncation: bool = True
    ) -> List[List[float]]:
        """
        Embed a list of music playlists.
        
        Args:
            playlists: List of MusicPlaylist objects
            input_type: "document" or "query"
            truncation: Whether to truncate long inputs
            
        Returns:
            List of embedding vectors (one per playlist)
        """
        texts = [self._playlist_to_text(playlist) for playlist in playlists]
        
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
            truncation=truncation
        )
        
        return result.embeddings
    
    def embed_playlist(self, playlist: MusicPlaylist) -> List[float]:
        """Embed a single playlist."""
        return self.embed_playlists([playlist])[0]
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query (e.g., "upbeat workout music from the 80s").
        
        Uses input_type="query" for optimal retrieval performance.
        """
        result = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query"
        )
        return result.embeddings[0]
    
    def find_similar_tracks(
        self,
        query_track: MusicTrack,
        candidate_tracks: List[MusicTrack],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find tracks similar to a query track using cosine similarity.
        
        Args:
            query_track: The track to find similarities to
            candidate_tracks: Pool of tracks to search through
            top_k: Number of similar tracks to return
            
        Returns:
            List of (track, similarity_score) tuples, sorted by similarity
        """
        import numpy as np
        
        # Embed query as document (since it's a track, not a search query)
        query_embedding = np.array(self.embed_track(query_track))
        
        # Embed candidates
        candidate_embeddings = np.array(self.embed_tracks(candidate_tracks))
        
        # Compute cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidates_norm = candidate_embeddings / np.linalg.norm(
            candidate_embeddings, axis=1, keepdims=True
        )
        
        similarities = np.dot(candidates_norm, query_norm)
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            (candidate_tracks[i], float(similarities[i]))
            for i in top_indices
        ]


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = MusicEmbeddingService()
    
    # Example: Create some tracks
    tracks = [
        MusicTrack(
            title="Bohemian Rhapsody",
            artist="Queen",
            album="A Night at the Opera",
            genre="Rock",
            year=1975,
            mood="epic, dramatic",
            tempo="moderate"
        ),
        MusicTrack(
            title="Billie Jean",
            artist="Michael Jackson",
            album="Thriller",
            genre="Pop",
            year=1982,
            mood="energetic, confident",
            tempo="fast"
        ),
        MusicTrack(
            title="Imagine",
            artist="John Lennon",
            album="Imagine",
            genre="Rock",
            year=1971,
            mood="peaceful, hopeful",
            tempo="slow"
        ),
    ]
    
    # Embed tracks
    print("Embedding tracks...")
    embeddings = service.embed_tracks(tracks)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Example: Embed a search query
    query = "upbeat classic rock from the 70s"
    print(f"\nEmbedding query: '{query}'")
    query_embedding = service.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Example: Find similar tracks
    print("\nFinding tracks similar to 'Bohemian Rhapsody'...")
    similar = service.find_similar_tracks(tracks[0], tracks, top_k=2)
    for track, score in similar:
        print(f"  - {track.title} by {track.artist} (similarity: {score:.3f})")
    
    # Example: Create and embed a playlist
    workout_playlist = MusicPlaylist(
        name="High Energy Workout",
        description="Upbeat tracks to power through your gym session",
        tracks=[tracks[1]],  # Billie Jean
        mood="energetic, motivating",
        occasion="workout"
    )
    
    print(f"\nEmbedding playlist: {workout_playlist.name}")
    playlist_embedding = service.embed_playlist(workout_playlist)
    print(f"Playlist embedding dimension: {len(playlist_embedding)}")
