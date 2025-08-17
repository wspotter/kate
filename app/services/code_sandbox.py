"""
Code execution sandbox using Docker for security isolation.
"""

import asyncio
from typing import Optional, Tuple


class CodeSandbox:
    """
    A simulated secure sandbox for executing untrusted code using Docker.
    """

    def __init__(self, image: str = "python:3.9-slim"):
        self._image = image
        self._container_id: Optional[str] = None

    async def __aenter__(self):
        """(Simulation) Starts a Docker container for code execution."""
        print(f"Starting Docker container with image: {self._image}...")
        await asyncio.sleep(2)  # Simulate container startup time
        self._container_id = "simulated_container_12345"
        print(f"Container started: {self._container_id}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """(Simulation) Stops and removes the Docker container."""
        if self._container_id:
            print(f"Stopping container: {self._container_id}...")
            await asyncio.sleep(1)
            print("Container stopped and removed.")
            self._container_id = None

    async def run(self, code: str, timeout: int = 10) -> Tuple[str, str, int]:
        """
        (Simulation) Executes code within the Docker container.

        Args:
            code: The code to execute.
            timeout: The maximum execution time in seconds.
        
        Returns:
            A tuple containing (stdout, stderr, return_code).
        """
        if not self._container_id:
            raise RuntimeError("Sandbox is not running.")

        print(f"Executing code in container {self._container_id} with a {timeout}s timeout...")
        await asyncio.sleep(2)  # Simulate execution time

        if "error" in code.lower():
            return "", "Simulated runtime error in sandbox.", 1
        
        return "Simulated output from sandboxed code.", "", 0


async def main():
    """Example usage of the CodeSandbox."""
    python_code = "print('Hello from the sandbox!')"
    
    async with CodeSandbox() as sandbox:
        stdout, stderr, return_code = await sandbox.run(python_code)
        
        print("\n--- Execution Results ---")
        print(f"Return Code: {return_code}")
        print(f"Stdout:\n{stdout}")
        print(f"Stderr:\n{stderr}")

if __name__ == "__main__":
    asyncio.run(main())