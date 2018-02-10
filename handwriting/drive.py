"""

Easily interact with Google Drive from Colaboratory.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


def drive_client():
    """get Drive client, authenticating if necessary"""
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive


def path_to_drive_file(drive, path):
    """get a Drive file given an absolute path"""
    prev_id = "root"
    for name in path.split("/"):
        dfile = drive.ListFile({'q': "title = '" + name + "' and '" + prev_id + "' in parents"}).GetList()[0]
        prev_id = dfile["id"]
    return dfile


def download_drive_file(drive_file, local_dir):
    """download a Drive file into a local directory"""
    drive_file.GetContentFile(os.path.join(local_dir, drive_file["title"]))


# TODO: upload file to Drive
