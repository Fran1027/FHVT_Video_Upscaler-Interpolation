import os
import sys

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_filename = "complete_python_code.txt"

    print(f"Searching for .py files in: {os.path.abspath(target_dir)}")
    count = 0

    with open(output_filename, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(target_dir):
            # Exclude virtual environments, interpreter caches, version control metadata, and binary folders
            if any(ignore in root for ignore in [".venv", "venv", "__pycache__", ".git", "bin"]):
                continue

            for file in files:
                # Exclude exporter utility scripts to prevent recursive concatenation
                if file.endswith(".py") and file != "export_py.py" and file != "export_kt.py":
                    count += 1
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, target_dir)
                    
                    out.write(f"{rel_path}\n")
                    out.write("-" * 59 + "\n\n")
                    try:
                        # Append file content with utf-8 encoding and fallback error tolerance
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            out.write(f.read())
                    except Exception as e:
                        out.write(f"Error reading file: {e}\n")
                    out.write("\n\n" + "-" * 59 + "\n")
                    out.write(f"end of file {rel_path}\n\n")
                    out.write("=" * 59 + "\n\n")

    print(f"Completed: Packaged {count} .py files into '{output_filename}'.")

if __name__ == "__main__":
    main()
