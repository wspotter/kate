"""
Service for handling code analysis, execution, and AI-powered insights.
"""

import asyncio
from typing import Any, Dict, Optional

from app.core.config import DatabaseSettings
from app.database.manager import DatabaseManager
from app.database.models import CodeExecution, ProcessingStatus


class CodeAnalysisService:
    """
    Manages code execution, syntax highlighting, and AI-driven analysis.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def execute_code(
        self,
        language: str,
        code: str,
        message_id: Optional[str] = None
    ) -> CodeExecution:
        """
        Executes a code snippet in a simulated sandboxed environment.

        Args:
            language: The programming language of the code.
            code: The code snippet to execute.
            message_id: The optional ID of the message associated with this execution.

        Returns:
            A CodeExecution object with the results of the execution.
        """
        async with self.db_manager.session() as session:
            # Create a record for the code execution
            code_execution = CodeExecution(
                message_id=message_id,
                language=language,
                code=code,
                status=ProcessingStatus.PROCESSING
            )
            session.add(code_execution)
            await session.commit()
            await session.refresh(code_execution)

            try:
                print(f"Executing {language} code in sandbox...")
                await asyncio.sleep(2)  # Simulate execution time

                # In a real implementation, this would use a secure sandbox
                stdout, stderr, return_code = self._run_in_sandbox(language, code)

                # Update the execution record with the results
                code_execution.status = ProcessingStatus.COMPLETED if return_code == 0 else ProcessingStatus.FAILED
                code_execution.stdout = stdout
                code_execution.stderr = stderr
                code_execution.return_code = return_code
                
                await session.commit()
                print("Code execution finished.")

            except Exception as e:
                print(f"Error executing code: {e}")
                code_execution.status = ProcessingStatus.FAILED
                code_execution.stderr = str(e)
                await session.commit()

            return code_execution

    def _run_in_sandbox(self, language: str, code: str) -> (str, str, int):
        """
        (Simulation) Runs code in a sandboxed environment.
        """
        if "error" in code.lower():
            return "", "Simulated runtime error.", 1
        return "Simulated output from code.", "", 0

    async def get_ai_insights(self, code_execution: CodeExecution) -> Dict[str, Any]:
        """
        (Simulation) Generates AI-powered insights for a given code execution.
        """
        print("Generating AI insights for code execution...")
        await asyncio.sleep(1)

        insights = {
            "summary": "The code appears to be a simulation of a simple function.",
            "suggestions": ["Consider adding more robust error handling.", "Add comments to improve readability."],
            "complexity": "Low"
        }
        return insights


async def main():
    """Example usage of the CodeAnalysisService."""
    settings = DatabaseSettings(url="sqlite+aiosqlite:///kate_test.db")
    db_manager = DatabaseManager(settings)
    await db_manager.initialize()
    await db_manager.create_tables()

    code_service = CodeAnalysisService(db_manager)
    
    # Execute a code snippet
    python_code = "def hello():\n    print('Hello, World!')\n\nhello()"
    execution_record = await code_service.execute_code("python", python_code)
    
    print("\nCode Execution Results:")
    print(f"  Status: {execution_record.status.value}")
    print(f"  Stdout: {execution_record.stdout}")
    
    # Get AI insights
    insights = await code_service.get_ai_insights(execution_record)
    print("\nAI-Powered Insights:")
    print(f"  Summary: {insights['summary']}")
    print(f"  Suggestions: {insights['suggestions']}")

if __name__ == "__main__":
    asyncio.run(main())