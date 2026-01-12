from simulation_manager import SimulationManager

# --- Main Execution ---
if __name__ == "__main__":
    
    print("--- Starting 3D Maze Robot Simulation ---")
    sim = None
    try:
        # 1. Initialize the Simulation Environment (loads maze and robot)
        sim = SimulationManager()
        
        # 2. Run the main loop (where control, perception, and SLAM operate)
        sim.run_simulation()

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        
    finally:
        # 3. Cleanly disconnect PyBullet
        if sim is not None:
            sim.disconnect()
            print("--- Simulation Ended and Disconnected ---")