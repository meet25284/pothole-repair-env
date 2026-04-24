from openenv.env import Env
from env import PotholeRepairEnv
from models import Action, ActionType


def verify():
    print("=== OpenEnv Compliance Check ===\n")

    # Check 1: Inheritance
    env = PotholeRepairEnv()
    assert isinstance(env, Env), "FAIL: Not inheriting from Env"
    print("✅ Inherits from openenv.env.Env")

    # Check 2: Required attributes
    assert hasattr(env, "name"), "FAIL: missing name"
    assert hasattr(env, "state_space"), "FAIL: missing state_space"
    assert hasattr(env, "action_space"), "FAIL: missing action_space"
    assert hasattr(env, "episode_max_length"), "FAIL: missing episode_max_length"
    print("✅ Has all required OpenEnv attributes")
    print(f"   name={env.name}")
    print(f"   episode_max_length={env.episode_max_length}")

    # Check 3: reset() works
    obs = env.reset()
    assert obs is not None, "FAIL: reset() returned None"
    print("✅ reset() works")

    # Check 4: step() works
    first_id = obs.potholes[0].id
    action = Action(
        action_type=ActionType.DISPATCH,
        pothole_id=first_id
    )
    result = env.step(action)
    assert result.reward is not None, "FAIL: step() no reward"
    assert isinstance(result.done, bool), "FAIL: done not bool"
    print("✅ step() works")
    print(f"   reward={result.reward} done={result.done}")

    # Check 5: state() works
    state = env.state()
    assert state is not None, "FAIL: state() returned None"
    print("✅ state() works")

    # Check 6: reward() method exists
    assert hasattr(env, "reward"), "FAIL: missing reward() method"
    print("✅ reward() method exists")

    # Check 7: print_info works
    print("\n--- print_info() output ---")
    env.print_info()

    # Check 8: all 3 tasks work
    for task in ["critical_repair", "budget_optimizer", "full_city_manager"]:
        e = PotholeRepairEnv(task_name=task)
        o = e.reset()
        assert len(o.potholes) > 0
        print(f"✅ Task '{task}' works ({len(o.potholes)} potholes)")

    print("\n=== ALL CHECKS PASSED ✅ ===")
    print("Your environment is OpenEnv compliant.")


if __name__ == "__main__":
    verify()
