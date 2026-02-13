import click
from expailens.dashboard.app import app

@click.command()
@click.option("--port", default=5000)
def run_dashboard(port):
    app.run(port=port)

if __name__ == "__main__":
    run_dashboard()