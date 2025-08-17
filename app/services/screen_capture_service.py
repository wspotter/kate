"""
Service for handling screen capture and screenshot analysis.
"""

import asyncio
from pathlib import Path
from typing import Optional

from PIL import ImageGrab


class ScreenCaptureService:
    """
    Manages screen capture and analysis functionality.
    """

    def capture_screen(self, output_dir: Path) -> Optional[Path]:
        """
        Captures the entire screen and saves it as an image file.

        Args:
            output_dir: The directory to save the screenshot in.

        Returns:
            The path to the captured screenshot, or None if capture fails.
        """
        try:
            screenshot = ImageGrab.grab()
            output_file = output_dir / "screenshot.png"
            screenshot.save(output_file, "PNG")
            print(f"Screenshot saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None

    async def analyze_screenshot(self, screenshot_path: Path) -> str:
        """
        (Simulation) Analyzes a screenshot to extract information.

        Args:
            screenshot_path: The path to the screenshot to analyze.

        Returns:
            A string containing the analysis of the screenshot.
        """
        print(f"Analyzing screenshot: {screenshot_path}...")
        await asyncio.sleep(2)  # Simulate analysis time
        
        analysis_result = (
            "Simulated analysis:\n"
            "- Identified a web browser window showing a stock chart.\n"
            "- The chart indicates a downward trend for the day.\n"
            "- A news headline is visible at the bottom of the screen."
        )
        
        print("Screenshot analysis complete.")
        return analysis_result


async def main():
    """Example usage of the ScreenCaptureService."""
    capture_service = ScreenCaptureService()
    output_directory = Path(".")
    
    # Capture the screen
    screenshot_file = capture_service.capture_screen(output_directory)
    
    if screenshot_file:
        # Analyze the screenshot
        analysis = await capture_service.analyze_screenshot(screenshot_file)
        print("\nScreenshot Analysis:")
        print(analysis)

if __name__ == "__main__":
    asyncio.run(main())