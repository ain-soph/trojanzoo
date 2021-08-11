#!/usr/bin/env python3

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

import io
import os
import pickle

from typing import TYPE_CHECKING
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import Resource
from googleapiclient.http import HttpRequest

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']


# To list folders
def listfolders(service: Resource, file_id: str, folder_path: str) -> None:
    results: dict = service.files().list(
        q=f"'{file_id}' in parents",
        fields="nextPageToken, files(id, name, mimeType)").execute()
    # logging.debug(folder)
    folder: list[dict[str, str]] = results.get('files', [])
    for item in folder:
        filepath = os.path.join(folder_path, item['name'])
        if str(item['mimeType']) == 'application/vnd.google-apps.folder':
            if not os.path.isdir(filepath):
                os.mkdir(filepath)
            listfolders(service, filepath)  # LOOP un-till the files are found
        elif not os.path.exists(filepath):
            print(filepath)
            downloadfiles(service, item['id'], item['name'], folder_path)


# To Download Files
def downloadfiles(service: Resource, dowid: str, name: str, dfilespath: str):
    request: HttpRequest = service.files().get_media(fileId=dowid)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100):3d}%.")
    with io.open(os.path.join(dfilespath, name), 'wb') as f:
        fh.seek(0)
        f.write(fh.read())


def download_folder(folder_id: str, output_path: str = './', token_path: str = 'token.pickle'):
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """

    creds: Credentials = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds: Credentials = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)  # credentials.json download from drive API
            creds = flow.run_local_server()
        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    # Call the Drive v3 API

    results: dict = service.files().list(q=f"'{folder_id}' in parents",
                                         fields="nextPageToken, files(id, name, mimeType)").execute()
    items = results.get('files', [])
    if not items:
        print('No files found.')
    else:
        print('Files:')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for item in items:
            filepath = os.path.join(output_path, item['name'])
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                if not os.path.exists(filepath):
                    os.mkdir(filepath)
                listfolders(service, item['id'], filepath)
            elif not os.path.exists(filepath):
                print(filepath)
                downloadfiles(service, item['id'], item['name'], output_path)
