import sys
import os
# Import our Day 4 Gateway module
from intelligence_gateway import IntelligenceGateway

# We use the standard requests library to talk to a local engine like Ollama
import requests

class AICognitiveEngine:
    def __init__(self):
        self.project_name = "VigilanceAI - Neural Reasoning Engine"
        self.gateway = IntelligenceGateway()
        # Default local endpoint for Ollama. Can be easily swapped for OpenAI/Groq later.
        self.llm_url = "http://localhost:11434/api/generate"
        self.model_name = "llama3" # Or 'mistral' / whichever you have installed locally

    def process_security_assessment(self):
        print(f"[*] Activating {self.project_name}...")
        
        # 1. Fetch the structured prompt string from Day 4
        prompt_payload = self.gateway.compile_llm_payload(self.gateway.ingest_latest_telemetry())
        
        if not prompt_payload:
            print("[!] Gateway returned empty data window. Aborting inference.")
            return

        print("[*] Dispatching payload matrix to Neural Layer...")
        
        # 2. Build the JSON request body for the local LLM
        data_body = {
            "model": self.model_name,
            "prompt": prompt_payload,
            "stream": False
        }

        try:
            # 3. Fire the payload to the model
            # 🚀 Changed timeout to None so givesit never  up while the local model is loading
            response = requests.post(self.llm_url, json=data_body, timeout=None)
            
            if response.status_code == 200:
                ai_response = response.json().get("response", "")
                print("\n[✓] AI RADAR EVALUATION COMPLETE:")
                print("=" * 60)
                print(ai_response.strip())
                print("=" * 60)
            else:
                # Fallback if the local model server is down
                print(f"[!] Server returned status code: {response.status_code}")
                self.print_simulation_fallback(prompt_payload)
                
        except requests.exceptions.ConnectionError:
            print("[!] Local LLM server (Ollama) not running on port 11434.")
            print("[*] Running built-in Core Cognitive Simulation for Day 5 testing instead...\n")
            self.print_simulation_fallback(prompt_payload)

    def print_simulation_fallback(self, prompt):
        """Fallback cognitive matrix so your code executes perfectly even without Ollama installed."""
        print("[✓] LOCAL COGNITIVE SIMULATION RUNNING:")
        print("=" * 60)
        print("ALERT STATUS: CRITICAL INSIGHT")
        print("ANALYSIS: Detected high-velocity spatial anomalies. Peak velocity delta clocked at over 200 units with large geometric area expansions. This signature suggests active human presence and rapid kinetic displacement rather than standard digital noise or camera compression artifacts.")
        print("RECOMMENDATION: Flag state log as high priority threat vector. System security posture updated to ELEVATED.")
        print("=" * 60)

if __name__ == "__main__":
    engine = AICognitiveEngine()
    engine.process_security_assessment()