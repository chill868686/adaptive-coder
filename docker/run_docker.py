import os
import pathlib
import signal
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
import docker
from docker import types

flags.DEFINE_string('log', 'running.log', 'Name of the log file.')
flags.DEFINE_string('file_path', None, 'Path to data|training file.')
#flags.DEFINE_string('model', None, 'Name of the en_decoding model.')     调整training.sh
flags.DEFINE_string('docker_image_name', 'b4k:1.0', 'Name of the Docker image.')
flags.DEFINE_enum(
    'coding_type', 'en_decoding',
    ['en_decoding', 'encoding', 'decoding','training'],
    'coding_type can be one of en_decoding|encoding|decoding|training')
flags.DEFINE_string(
    'gpu_devices', 'all',
    'Comma separated list of devices to pass to NVIDIA_VISIBLE_DEVICES.')
flags.DEFINE_string(
    'docker_user', f'{os.geteuid()}:{os.getegid()}',
    'UID:GID with which to run the Docker container. The output directories '
    'will be owned by this user:group. By default, this is the current user. '
    'Valid options are: uid or uid:gid, non-numeric values are not recognised '
    'by Docker unless that user has been created within the container.')
FLAGS = flags.FLAGS
_ROOT_MOUNT_DIRECTORY = '/mnt/'

def _create_mount(mount_name: str, path: str) -> Tuple[types.Mount, str]:
    '''Create a mount point for each file and directory used by the model.'''
    path = pathlib.Path(path).absolute()
    target_path = pathlib.Path(_ROOT_MOUNT_DIRECTORY, mount_name)

    if path.is_dir():
        source_path = path
        mounted_path = target_path
    else:
        source_path = path.parent
        mounted_path = pathlib.Path(target_path, path.name)
    if not source_path.exists():
        raise ValueError(f'Failed to find source directory "{source_path}" to '
                     'mount in Docker container.')
    logging.info('Mounting %s -> %s', source_path, target_path)
    mount = types.Mount(target=str(target_path), source=str(source_path),
                      type='bind', read_only=False)
    return mount, str(mounted_path)

def main(argv):

    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    mounts = []
    #volumes = {}
    command_args = []

    adaptive_coder_path = pathlib.Path(__file__).parent.parent

    paths = [
        ('adaptive_coder_path', str(adaptive_coder_path)),
        ('file_path', FLAGS.file_path),
    ]

    for name, path in paths:
        if path:
            mount, target_path = _create_mount(name, path)
            mounts.append(mount)
            command_args.append(f'--{name}={target_path}')

    #['en_decoding', 'encoding', 'decoding']
    if FLAGS.coding_type == 'en_decoding':
        entrypoint = 'conversion.sh'
    elif FLAGS.coding_type == 'encoding':
        entrypoint = 'conversion.sh'
    elif FLAGS.coding_type == 'decoding':
        entrypoint = 'conversion.sh'
    elif FLAGS.coding_type == 'training':
        #model name is are consistent with training file.
        entrypoint = 'training.sh'

    entrypoint = os.path.join('/mnt/adaptive_coder_path',entrypoint)
    command_args.extend([
        f'--coding_type={FLAGS.coding_type}',
        f'--log={FLAGS.log}'
    ])

    client = docker.from_env()
    device_requests = [docker.types.DeviceRequest(driver='nvidia', capabilities=[['gpu']])]

    container = client.containers.run(
        image=FLAGS.docker_image_name,
        command=command_args,
        device_requests=device_requests,
        entrypoint = entrypoint,
        remove=True,
        detach=True,
        mounts=mounts,
        user=FLAGS.docker_user,
        environment={
            'NVIDIA_VISIBLE_DEVICES': FLAGS.gpu_devices,
        })

    # Add signal handler to ensure CTRL+C also stops the running container.
    signal.signal(signal.SIGINT, lambda unused_sig, unused_frame: container.kill())

    for line in container.logs(stream=True):
        print(line.strip().decode('utf-8'))

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'file_path'
  ])
  app.run(main)
