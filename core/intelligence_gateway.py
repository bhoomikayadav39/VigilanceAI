import json
import os
import sys
from datetime import datetime

class IntelligenceGateway:
    def __init__(self):
        self.log_path = "logs/motion_telemetry.json"
        
    def ingest_latest_telemetry(self):
        """Reads local telemetry state data and structures it safely for LLM evaluation."""
        if not os.path.exists(self.log_path):
            return None
            
        try:
            with open(self.log_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return None
                return json.loads(content)
        except Exception as e:
            return None

    def compile_llm_payload(self, events):
        """Transforms raw numerical vectors into a semantic prompt for the LLM layer."""
        window_events = events[-5:] if len(events) >= 5 else events
        total_events = len(window_events)
        
        max_velocity = max(e["metrics"]["calculated_velocity_delta"] for e in window_events)
        avg_area = sum(e["metrics"]["spatial_area_pixels"] for e in window_events) / total_events
        
        heuristic_severity = "LOW"
        if max_velocity > 40.0 or avg_area > 15000:
            heuristic_severity = "HIGH"
            
        prompt_context = f"""
[SYSTEM ALERT MATRIX - UNPROCESSED TELEMETRY LOG]
Timestamp Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Sensor Type: OpenCV Edge Temporal Motion Frame-Differencer
Raw Event Count in Window: {total_events}
Peak Velocity Delta Observed: {max_velocity:.2f}
Average Geometric Mass Area: {avg_area:.1f} pixels
Initial Heuristic Threat Level: {heuristic_severity}

Recent Activity Timeline Logs:
"""
        for i, event in enumerate(window_events, 1):
            prompt_context += f" - Event #{i} [{event['timestamp']}]: Area={event['metrics']['spatial_area_pixels']}px, Velocity Delta={event['metrics']['calculated_velocity_delta']:.2f}\n"
            
        prompt_context += """
INSTRUCTION FOR AI COGNITIVE LAYER:
Analyze the structural anomalies above. Determine if this sequence indicates high-velocity tampering, a breach, or simple ambient background shifts. Return a clean, structured analysis payload summarizing the real-time operational security posture.
"""
        return prompt_context

    def stage_data(self):
        events = self.ingest_latest_telemetry()
        if events:
            llm_prompt = self.compile_llm_payload(events)
            print("=" * 60)
            print(llm_prompt.strip())
            print("=" * 60)
            return llm_prompt
        return None

if __name__ == "__main__":
    gateway = IntelligenceGateway()
    gateway.stage_data()