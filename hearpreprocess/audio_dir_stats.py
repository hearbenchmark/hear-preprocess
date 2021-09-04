import click

import hearpreprocess.util.audio as audio_util


@click.command()
@click.argument("in-dir")
@click.argument("out-file")
def audio_dir_stats(in_dir: str, out_file: str):
    """Command line click endpoint to get audio directory stats"""
    audio_util.get_audio_dir_stats(in_dir, out_file)


if __name__ == "__main__":
    audio_dir_stats()
