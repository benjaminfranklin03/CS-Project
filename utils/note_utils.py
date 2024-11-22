# ===========================================================
# Standard Library Imports
# ===========================================================
import os
import logging
import cloudpickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# ===========================================================
# Third-Party Imports
# ===========================================================
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation

# ===========================================================
# Configure Logging
# ===========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================
# Constant
# ===========================================================
EMBEDDINGS_CACHE = 'cache/embeddings_cache.pkl'

# ===========================================================
# Data Class
# ===========================================================
@dataclass
class Note:
    """
    Data class representing a note with its content, title, and metadata.
    """
    id: str
    content: str
    title: str
    created_at: datetime
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    summary: Optional[str] = None #storing summary
    is_centroid: bool = False #for identifying centroids

# ===========================================================
# Note Embedding System Class
# ===========================================================
class NoteEmbeddingSystem:
    """
    A system to manage notes and compute semantic embeddings for clustering and searching.
    """
    def __init__(self, 
                model_name: str = 'all-MiniLM-L6-v2',
                summarizer_name: str = 'sshleifer/distilbart-cnn-12-6',
                device: Optional[str] = None):
        """
        Initialize the NoteEmbeddingSystem with a SentenceTransformer model.
        
        Args:
            cache_dir (str): Directory to cache embeddings.
            model_name (str): Name of the SentenceTransformer model to use.
            device (Optional[str]): Device to run the model on ('cuda' or 'cpu').
        """
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(EMBEDDINGS_CACHE), exist_ok=True)

        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load the SentenceTransformer model
        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

        # Load the summarization model
        try:
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_name, use_fast=True) #use_fast uses faster implemenation of tokenizer
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name).to(self.device)
            logger.info(f"Loaded summarization model: {summarizer_name}")
        except Exception as e:
            logger.error(f"Failed to load summarization model '{summarizer_name}': {e}")
            raise

        # Initialize notes and embeddings
        self.notes: Dict[str, Note] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.pca = PCA(n_components=2)

        # Load cached embeddings if available
        self._load_cache()

    # =======================================================
    # Note Management Methods
    # =======================================================

    def add_note(self, note_id: str, content: str, title: str) -> Note:
        """
        Add a new note, generate its semantic embedding, and create a summary.
        """
        if any(note.title == title for note in self.notes.values()):
            raise ValueError(f"A note with the title '{title}' already exists. Please use a unique title.")

        if note_id in self.notes:
            raise ValueError(f"Note with ID {note_id} already exists")

        try:
            # Generate semantic embedding
            embedding = self._generate_embedding(content)

            # Generate summary
            summary = self._generate_summary(content)

            # Create Note instance
            note = Note(
                id=note_id,
                content=content,
                title=title,
                created_at=datetime.now(),
                embedding=embedding,
                summary=summary  # Store the summary
            )
            self.notes[note_id] = note
            logger.info(f"Added note: {note_id}")

            # Update embeddings matrix
            self._update_embeddings_matrix()

            # Save to cache
            self._save_cache()

            return note
        except Exception as e:
            logger.error(f"Failed to add note '{note_id}': {e}")
            raise

    def delete_note(self, note_id: str) -> bool:
        """
        Delete a note by its ID and update related states.
        
        Args:
            note_id (str): The ID of the note to delete.
        
        Returns:
            bool: True if the note was deleted successfully, False otherwise.
        """
        if note_id not in self.notes:
            logger.warning(f"Attempted to delete a non-existent note with ID: {note_id}")
            return False

        try:
            # Remove the note
            del self.notes[note_id]
            logger.info(f"Deleted note with ID: {note_id}")

            # Update embeddings matrix
            self._update_embeddings_matrix()
            logger.info("Updated embeddings matrix after deletion.")

            # Save updated state to cache
            self._save_cache()
            logger.info("Saved updated state to cache.")

            return True
        except Exception as e:
            logger.error(f"Error while deleting note with ID {note_id}: {e}")
            return False

    def _generate_summary(self, text: str, max_length: int = 120, min_length: int = 30) -> str:
        """
        Generate a summary for the given text using the summarization model that takes in the first 1024 tokens as input.
        """
        try:
            inputs = self.summarizer_tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=1024  
            ).to(self.device)
            
            #Using FP16 for faster inference
            with torch.amp.autocast('cuda'):
            # Generate summary IDs 
                summary_ids = self.summarizer_model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=2, #low beam search breadth for faster inference
                    early_stopping=True #stop at <EOS> token
                )

            # Decode the generated summary
            summary = self.summarizer_tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    # =======================================================
    # Embedding and Clustering Methods
    # =======================================================

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding using the SentenceTransformer model.
        
        Args:
            text (str): The text to embed.
        
        Returns:
            np.ndarray: The embedding vector.
        """
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _update_embeddings_matrix(self):
        """
        Update the matrix of all note embeddings.
        """
        if self.notes:
            self.embeddings_matrix = np.vstack([
                note.embedding for note in self.notes.values()
            ])
            logger.info("Updated embeddings matrix.")
        else:
            self.embeddings_matrix = None
            logger.info("Embeddings matrix is empty.")

    def cluster_notes(self) -> Dict[int, List[str]]:
        """
        Cluster all notes using Affinity Propagation based on semantic similarity.
        
        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster IDs to lists of note IDs.
        """
        if not self.notes:
            logger.info("No notes to cluster.")
            return {}

        try:
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(self.embeddings_matrix)
            logger.info("Computed cosine similarity matrix for clustering.")

            # Perform Affinity Propagation clustering
            clustering = AffinityPropagation(affinity='precomputed', random_state=42)
            cluster_labels = clustering.fit_predict(similarity_matrix)
            logger.info("Performed Affinity Propagation clustering.")

            # Get cluster centers (exemplars)
            cluster_centers_indices = clustering.cluster_centers_indices_

            # Map indices to note IDs
            note_ids = list(self.notes.keys())
            centroids = [note_ids[index] for index in cluster_centers_indices]

            # Assign cluster IDs and centroid status to notes
            for idx, (note_id, cluster_id) in enumerate(zip(note_ids, cluster_labels)):
                note = self.notes[note_id]
                note.cluster_id = int(cluster_id)
                note.is_centroid = note_id in centroids  # Assign centroid status

            # Organize notes into clusters
            clusters: Dict[int, List[str]] = {}
            for note_id, cluster_id in zip(note_ids, cluster_labels):
                clusters.setdefault(cluster_id, []).append(note_id)

            # Store centroids in the clusters dictionary
            self.centroids = centroids  # Keep track of centroid note IDs
            logger.info(f"Formed {len(clusters)} clusters with centroids.")
            return clusters
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            return {}

    def get_2d_visualization(self) -> Dict[str, Tuple[float, float]]:
        """
        Generate 2D coordinates for visualizing notes using PCA.
        
        Returns:
            Dict[str, Tuple[float, float]]: Mapping of note IDs to (x, y) coordinates.
        """
        if not self.notes:
            logger.info("No notes to visualize.")
            return {}

        try:
            coords_2d = self.pca.fit_transform(self.embeddings_matrix)
            visualization = {
                note_id: (float(x), float(y))
                for note_id, (x, y) in zip(self.notes.keys(), coords_2d)
            }
            logger.info("Generated 2D visualization coordinates.")
            return visualization
        except Exception as e:
            logger.error(f"Error in generating 2D visualization: {e}")
            return {}

    # =======================================================
    # Similarity and Search Methods
    # =======================================================

    def get_similar_notes(self, note_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar notes to a given note based on semantic similarity.
        
        Args:
            note_id (str): ID of the reference note.
            top_k (int): Number of similar notes to retrieve.
        
        Returns:
            List[Tuple[str, float]]: List of tuples containing (note_id, similarity_score).
        """
        if note_id not in self.notes:
            logger.warning(f"Note ID '{note_id}' not found.")
            return []

        if len(self.notes) < 2:
            logger.info("Not enough notes to perform similarity search.")
            return []

        try:
            query_embedding = self.notes[note_id].embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

            # Exclude the query note itself
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            note_ids = list(self.notes.keys())

            similar_notes = [(note_ids[idx], similarities[idx]) for idx in similar_indices]
            logger.info(f"Found {len(similar_notes)} similar notes for '{note_id}'.")
            return similar_notes
        except Exception as e:
            logger.error(f"Error in getting similar notes: {e}")
            return []

    def semantic_search(self, 
                        query: str, 
                        top_k: int = 5,
                        threshold: float = 0.1) -> List[Tuple[str, float, str]]:
        """
        Search notes using semantic similarity with explanations.
        
        Args:
            query (str): Search query.
            top_k (int): Number of results to return.
            threshold (float): Minimum similarity threshold.
        
        Returns:
            List[Tuple[str, float, str]]: List of tuples containing (note_id, similarity_score, explanation).
        """
        if not self.notes:
            logger.info("No notes available for semantic search.")
            return []

        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query).reshape(1, -1)

            # Calculate semantic similarities
            similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

            # Get top matches above threshold
            matches = []
            note_ids = list(self.notes.keys())
            sorted_indices = np.argsort(similarities)[::-1]

            for idx in sorted_indices:
                sim_score = similarities[idx]
                if sim_score < threshold:
                    break
                note_id = note_ids[idx]
                note = self.notes[note_id]
                explanation = self._generate_similarity_explanation(
                    query, note.content, sim_score
                )
                matches.append((note_id, sim_score, explanation))
                if len(matches) >= top_k:
                    break

            logger.info(f"Found {len(matches)} matches for query.")
            return matches
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    def _generate_similarity_explanation(self, 
                                         query: str, 
                                         note_content: str, 
                                         similarity: float) -> str:
        """
        Generate human-readable explanation of semantic similarity.
        
        Args:
            query (str): The search query.
            note_content (str): The content of the note.
            similarity (float): The similarity score.
        
        Returns:
            str: An explanation of the similarity.
        """
        if similarity > 0.9:
            return "Very strong semantic match - concepts are nearly identical."
        elif similarity > 0.8:
            return "Strong semantic relationship - discusses the same key concepts."
        elif similarity > 0.7:
            return "Moderate semantic overlap - shares some important concepts."
        elif similarity > 0.6:
            return "Weak semantic connection - tangentially related concepts."
        else:
            return "Minimal semantic similarity - concepts are mostly different."

    # =======================================================
    # Cache Management Methods
    # =======================================================

    def _save_cache(self):
        """
        Save embeddings and notes to cache.
        """
        try:
            with open(EMBEDDINGS_CACHE, 'wb') as f:
                cloudpickle.dump({
                    'notes': self.notes,
                    'embeddings_matrix': self.embeddings_matrix
                }, f)
            logger.info("Saved cache successfully.")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_cache(self):
        """
        Load embeddings and notes from cache.
        """
        try:
            if os.path.exists(EMBEDDINGS_CACHE):
                with open(EMBEDDINGS_CACHE, 'rb') as f:
                    cache_data = cloudpickle.load(f)
                    self.notes = cache_data['notes']
                    self.embeddings_matrix = cache_data['embeddings_matrix']
                logger.info("Loaded cache successfully.")
            else:
                logger.info("No cache found. Starting fresh.")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")