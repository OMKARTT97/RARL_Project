# run_sweep.py - Script to initialize and run W&B sweeps

import wandb
import subprocess
import sys
import argparse

def create_and_run_sweep(project_name="cup_exp_1", sweep_config_path="sweep_config.yaml", count=20, parallel_agents=1):
    """
    Initialize a W&B sweep and run agents.
    
    Args:
        project_name: W&B project name
        sweep_config_path: Path to sweep configuration YAML file
        count: Total number of runs to execute
        parallel_agents: Number of parallel agents to spawn
    """
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config_path,
        project=project_name
    )
    
    print(f"✅ Sweep initialized with ID: {sweep_id}")
    print(f"📊 View results at: https://wandb.ai/[your-entity]/{project_name}/sweeps/{sweep_id.split('/')[-1]}")
    
    # Calculate runs per agent
    runs_per_agent = count // parallel_agents
    remainder = count % parallel_agents
    
    print(f"\n🚀 Spawning {parallel_agents} agents...")
    
    agents = []
    for i in range(parallel_agents):
        # Last agent takes remainder
        agent_count = runs_per_agent + (remainder if i == parallel_agents - 1 else 0)
        
        cmd = [
            sys.executable, "-m", "wandb", "agent",
            "--count", str(agent_count),
            sweep_id
        ]
        
        print(f"   Agent {i+1}/{parallel_agents}: {agent_count} runs")
        proc = subprocess.Popen(cmd)
        agents.append(proc)
    
    # Wait for all agents to complete
    print("\n⏳ Running sweep agents... Press Ctrl+C to stop.")
    try:
        for proc in agents:
            proc.wait()
    except KeyboardInterrupt:
        print("\n⚠️  Stopping sweep agents...")
        for proc in agents:
            proc.terminate()
    
    print(f"\n✅ Sweep complete! View results at https://wandb.ai")


def run_sweep_inline(project_name="tnn-ac-sweep", sweep_config_path="sweep_config.yaml", count=20):
    """
    Run sweep using inline Python API (useful for notebooks).
    """
    import yaml
    
    # Load config
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name
    )
    
    print(f"✅ Sweep initialized: {sweep_id}")
    
    # Import training function
    from region_tnn_sweeps import train_agent
    
    # Run agents
    wandb.agent(sweep_id, function=train_agent, count=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run W&B hyperparameter sweep for TNN Actor-Critic")
    parser.add_argument("--project", default="tnn-ac-sweep", help="W&B project name")
    parser.add_argument("--config", default="sweep_config.yaml", help="Path to sweep config YAML")
    parser.add_argument("--count", type=int, default=20, help="Total number of runs")
    parser.add_argument("--agents", type=int, default=1, help="Number of parallel agents")
    parser.add_argument("--inline", action="store_true", help="Use inline Python API instead of CLI")
    
    args = parser.parse_args()
    
    if args.inline:
        run_sweep_inline(args.project, args.config, args.count)
    else:
        create_and_run_sweep(args.project, args.config, args.count, args.agents)
