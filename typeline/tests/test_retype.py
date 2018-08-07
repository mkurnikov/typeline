from pathlib import Path

if __name__ == '__main__':
    from typeline import retype

    retype.Config.incremental = True

    BASE_DIR = Path(__file__).parent
    retype.retype_file(BASE_DIR / 'myclass.py', BASE_DIR, BASE_DIR)
