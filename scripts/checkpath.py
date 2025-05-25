



from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

from isaaclab.utils.assets import check_file_path
from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR


def main():
    _path = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac/Robots/Unitree/G1/G1_with_hand/g1_29dof_with_hand_rev_1_0.usd"
    print("Checking path: ", _path)
    print(check_file_path(_path))
    print("Done")


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()

