import time
import asyncio
from typing import Type, Dict, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from fastmcp.client import Client

class WaitCalcInput(BaseModel):
    """Input schema for WaitCalcTool"""
    calculation_ids: List[str] = Field(..., description="List of calculation IDs to check")

class WaitCalcTool(BaseTool):
    mcp_url: str = "http://localhost:8933/mcp"
    args_schema: Type[BaseModel] = WaitCalcInput

    def __init__(self, mcp_url: str):
        super().__init__(
            name="wait_calculations",
            description="Check the status of calculation tasks and return the results"
        )
        self.mcp_url = mcp_url

    async def _check_status(self, calculation_ids: List[str]) -> Dict[str, Any]:
        async with Client(self.mcp_url) as client:
            # call tool
            tool_result = await client.call_tool("check_calculation_status", {"calculation_ids": calculation_ids})
        if tool_result.data is None:
            return {"error": "No result from check_calculation_status"}
        else:
            return tool_result.data

    def _run(self,
             calculation_ids: List[str]) -> Dict[str, Any]:
        """
        Check the status of calculation tasks and return the results

        Args:
            calculation_ids: List of calculation IDs to check

        Returns:
            Dictionary containing the status and results of each calculation task, formatted as:
            {
                calculation_id: {
                    "slurm_id": "12345",
                    "calc_type": "relaxation",
                    "calculate_path": "/path/to/calculation",
                    "status": "running/completed/failed/error",
                    ... other result data
                }
            }
        """
        if not calculation_ids:
            return {}

        print(f"Starting to monitor calculation status, calculation IDs: {calculation_ids}")

        # Track completed and final results
        completed_results = {}
        pending_calc_ids = calculation_ids.copy()

        while pending_calc_ids:
            try:
                # Only check calculation tasks that have not finished yet
                status_result = asyncio.run(self._check_status(pending_calc_ids))

                if "error" in status_result:
                    print(f"Error while checking status: {status_result['error']}")
                    return status_result

                # Determine which calculations have finished and remove them from the pending list
                newly_completed = []
                for calc_id in pending_calc_ids.copy():
                    if calc_id in status_result:
                        status = status_result[calc_id].get("status", "unknown")
                        if status in ["completed", "failed", "cancelled", "unknown", "timeout"]:
                            # Task finished, save the result and remove it from the pending list
                            completed_results[calc_id] = status_result[calc_id]
                            newly_completed.append(calc_id)
                            pending_calc_ids.remove(calc_id)
                    else:
                        # If a calculation ID is not in the results, keep it in the pending list for now
                        pass

                # Summarize the current status
                running_count = len(pending_calc_ids)
                completed_count = len([r for r in completed_results.values() if r.get("status") == "completed"])
                failed_count = len([r for r in completed_results.values() if r.get("status") in ["failed", "unknown"]])

                if newly_completed:
                    print(f"Newly completed tasks: {newly_completed}")

                print(f"Status check result: running {running_count}, completed {completed_count}, failed {failed_count}")

                # If all calculations have finished, return the results
                if not pending_calc_ids:
                    print("All calculation tasks have completed")
                    return completed_results

                # Wait 30 seconds before checking again
                print(f"{len(pending_calc_ids)} task(s) still pending, waiting 30 seconds before checking again...")
                time.sleep(30)

            except Exception as e:
                print(f"An error occurred while monitoring: {str(e)}")
                return {"error": f"An error occurred while monitoring: {str(e)}"}

        # This point should not be reached in practice, but a default return value satisfies the linter
        return completed_results
