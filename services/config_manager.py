import os
import re
import logging
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages LLM API connections and PatternFinder configuration parsing."""
    def __init__(self):
        self.config_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")
        self.agent_config = {}
        self.model_credentials = {}
        load_dotenv()
        self.load_configuration()

    def load_configuration(self):
        """Load variables directly from config.py."""
        self.agent_config = {
            "min_win_rate": 0.40,
            "min_profit_factor": 1.4,
            "min_trades": 10,
            "last_selected_model_id": "deepseek-chat"
        }
        
        if os.path.exists(self.config_py_path):
            try:
                with open(self.config_py_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Regex match variables
                wr_match = re.search(r'MIN_WIN_RATE\s*=\s*([\d.]+)', content)
                pf_match = re.search(r'MIN_PROFIT_FACTOR\s*=\s*([\d.]+)', content)
                mt_match = re.search(r'MIN_TRADES\s*=\s*(\d+)', content)
                
                if wr_match: self.agent_config["min_win_rate"] = float(wr_match.group(1))
                if pf_match: self.agent_config["min_profit_factor"] = float(pf_match.group(1))
                if mt_match: self.agent_config["min_trades"] = int(mt_match.group(1))
                
            except Exception as e:
                logger.error(f"Error reading config.py: {e}")

        # Default model settings — ALL keys from env vars or user input, NEVER hardcoded
        self.models_data = [
            {"display_name": "GPT-5.2",              "model_id": "gpt-5.2-2025-12-11",             "api_key_env": "OPENAI_API_KEY"},
            {"display_name": "GPT-5 Mini",            "model_id": "gpt-5-mini-2025-08-07",          "api_key_env": "OPENAI_API_KEY"},
            {"display_name": "GPT-5 Nano",            "model_id": "gpt-5-nano-2025-08-07",          "api_key_env": "OPENAI_API_KEY"},
            {"display_name": "Claude Opus 4.6",       "model_id": "claude-opus-4-6",                "api_key_env": "ANTHROPIC_API_KEY"},
            {"display_name": "Claude Sonnet 4.6",     "model_id": "claude-sonnet-4-6",              "api_key_env": "ANTHROPIC_API_KEY"},
            {"display_name": "DeepSeek Chat",            "model_id": "deepseek-chat",                  "api_key_env": "DEEPSEEK_API_KEY"},
            {"display_name": "DeepSeek R1",            "model_id": "deepseek-reasoner",              "api_key_env": "DEEPSEEK_API_KEY"},
            {"display_name": "Grok 4.1 Reasoning",    "model_id": "grok-4-1-fast-reasoning",        "api_key_env": "XAI_API_KEY"},
            {"display_name": "Grok 4.1",               "model_id": "grok-4-1-fast-non-reasoning",    "api_key_env": "XAI_API_KEY"},
        ]
        
        for model in self.models_data:
            model_id = model.get("model_id")
            api_key_env = model.get("api_key_env")
            if api_key_env:
                self.model_credentials[model_id] = {
                    "api_key": os.getenv(api_key_env, "")
                }

    def get_agent_config(self):
        return self.agent_config

    def save_agent_config(self, new_config):
        """Rewrite thresholds back to config.py."""
        self.agent_config.update(new_config)
        if not os.path.exists(self.config_py_path):
            return
            
        try:
            with open(self.config_py_path, 'r', encoding='utf-8') as f:
                 lines = f.readlines()
            
            for i, line in enumerate(lines):
                if line.startswith("MIN_WIN_RATE"):
                    lines[i] = f"MIN_WIN_RATE      = {self.agent_config['min_win_rate']}\n"
                elif line.startswith("MIN_PROFIT_FACTOR"):
                    lines[i] = f"MIN_PROFIT_FACTOR = {self.agent_config['min_profit_factor']}\n"
                elif line.startswith("MIN_TRADES"):
                    lines[i] = f"MIN_TRADES        = {self.agent_config['min_trades']}\n"
                    
            with open(self.config_py_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        except Exception as e:
             logger.error(f"Could not save config to config.py: {e}")

    def save_env_and_credentials(self, env_updates, model_credentials):
        self.model_credentials = model_credentials
        for model_spec in self.models_data:
            model_id = model_spec.get('model_id')
            api_key_env = model_spec.get('api_key_env')
            if model_id in self.model_credentials and api_key_env:
                env_updates[api_key_env] = self.model_credentials[model_id].get('api_key', '')
        self._update_env_file(env_updates)

    def _update_env_file(self, updates: dict):
        env_file = find_dotenv()
        if not env_file: env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        try:
            lines = []
            if os.path.exists(env_file):
                with open(env_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            updated_keys = set()
            for i, line in enumerate(lines):
                match = re.match(r'^\s*([a-zA-Z0-9_]+)\s*=', line)
                if match:
                    key = match.group(1)
                    if key in updates:
                        lines[i] = f'{key}="{updates[key]}"\n'
                        updated_keys.add(key)
            for key, value in updates.items():
                if key not in updated_keys:
                    lines.append(f'{key}="{value}"\n')
            with open(env_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            logger.error(f"Error updating .env file: {e}")