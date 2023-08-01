from invoke import task


@task
def lint(command):
    print("Running Black...")
    command.run("black .")
    print("Running isort...")
    command.run("isort .")
    print("Running autoflake...")
    command.run("autoflake --remove-all-unused-imports --recursive --in-place . --exclude=__init__.py")
    print("Running Pylint...")
    command.run("pylint **/*.py")  # Adjust the path as needed
