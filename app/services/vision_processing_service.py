"""
Service for processing and analyzing visual content using multi-modal AI models.

This service handles image loading, pre-processing, analysis through vision models,
and storing the results in the database.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import DatabaseSettings
from app.database.manager import DatabaseManager
from app.database.models import ContentType, MediaContent, ProcessingStatus, VisionAnalysis


class VisionProcessingService:
    """
    Service for handling image analysis and visual content understanding.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def process_image(self, media_content_id: str, model_name: str = "gemini-2.5-pro") -> Optional[VisionAnalysis]:
        """

        Process an image, generate analysis using a vision model, and store the results.

        Args:
            media_content_id: The ID of the MediaContent to process.
            model_name: The name of the vision model to use for analysis.
        
        Returns:
            The created VisionAnalysis object or None if processing fails.
        """
        async with self.db_manager.session() as session:
            media_content = await session.get(MediaContent, media_content_id)
            if not media_content or media_content.content_type != ContentType.IMAGE:
                print(f"Error: MediaContent {media_content_id} not found or is not an image.")
                return None

            await self._update_status(session, media_content, ProcessingStatus.PROCESSING)

            try:
                # Simulate analysis with a multi-modal model
                analysis_results = await self._analyze_with_vision_model(media_content.file_path, model_name)

                # Create and store the vision analysis record
                vision_analysis = VisionAnalysis(
                    media_content_id=media_content.id,
                    model_name=model_name,
                    description=analysis_results.get("description"),
                    objects_detected=analysis_results.get("objects"),
                    text_extracted=analysis_results.get("ocr_text"),
                    labels=analysis_results.get("labels"),
                    confidence_scores=analysis_results.get("confidence"),
                    analysis_type="general" 
                )
                session.add(vision_analysis)
                
                # Update the MediaContent record with extracted info
                media_content.extracted_text = analysis_results.get("ocr_text")
                media_content.analysis_results = analysis_results
                await self._update_status(session, media_content, ProcessingStatus.COMPLETED)

                await session.commit()
                await session.refresh(vision_analysis)
                return vision_analysis

            except Exception as e:
                print(f"Error processing image {media_content_id}: {e}")
                await self._update_status(session, media_content, ProcessingStatus.FAILED)
                await session.commit()
                return None

    async def _analyze_with_vision_model(self, image_path: str, model_name: str) -> Dict[str, Any]:
        """
        (Simulation) Analyzes an image with a vision model.
        In a real implementation, this would call the multi-modal model provider.
        """
        print(f"Analyzing image {image_path} with {model_name}...")
        await asyncio.sleep(2)  # Simulate network latency and processing time

        # Simulate getting image metadata
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            width, height = 0, 0

        # Mocked analysis results
        mock_analysis = {
            "description": f"A detailed description of the image content at path {image_path}.",
            "objects": [
                {"box": [10, 20, 50, 60], "label": "Cat", "score": 0.95},
                {"box": [60, 30, 100, 80], "label": "Table", "score": 0.88},
            ],
            "ocr_text": "Sample text extracted from the image.",
            "labels": ["Pet", "Furniture", "Indoors"],
            "confidence": {"Pet": 0.96, "Furniture": 0.91, "Indoors": 0.99},
            "image_dimensions": {"width": width, "height": height}
        }
        print("Analysis complete.")
        return mock_analysis

    async def _update_status(self, session: AsyncSession, media_content: MediaContent, status: ProcessingStatus):
        """Helper to update the processing status of a MediaContent object."""
        media_content.processing_status = status
        session.add(media_content)

async def main():
    """Example usage of the VisionProcessingService."""
    settings = DatabaseSettings(url="sqlite+aiosqlite:///kate_test.db")
    db_manager = DatabaseManager(settings)
    await db_manager.initialize()
    await db_manager.create_tables()
    
    vision_service = VisionProcessingService(db_manager)

    # Create a dummy MediaContent record for testing
    async with db_manager.session() as session:
        # Create a dummy image file
        dummy_image_path = Path("dummy_image.png")
        if not dummy_image_path.exists():
            try:
                Image.new('RGB', (100, 100), color = 'red').save(dummy_image_path)
            except Exception as e:
                print(f"Could not create dummy image: {e}")
                return

        new_media = MediaContent(
            filename="test_image.png",
            file_path=str(dummy_image_path.resolve()),
            content_type=ContentType.IMAGE,
            file_size=1024,
            mime_type="image/png"
        )
        session.add(new_media)
        await session.commit()
        await session.refresh(new_media)
        media_id = new_media.id

    print(f"Created dummy MediaContent with ID: {media_id}")
    
    # Process the image
    analysis = await vision_service.process_image(media_id)
    if analysis:
        print("\nVision Analysis Results:")
        print(f"  Description: {analysis.description}")
        print(f"  Objects Detected: {analysis.objects_detected}")
        print(f"  Extracted Text: {analysis.text_extracted}")
    else:
        print("\nVision analysis failed.")

if __name__ == "__main__":
    asyncio.run(main())