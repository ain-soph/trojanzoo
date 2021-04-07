from __future__ import print_function
import pickle
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
import io
import argparse


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']


# To list folders
def listfolders(service, filid, des):
    results = service.files().list(
        q="\'" + filid + "\'" + " in parents",
        fields="nextPageToken, files(id, name, mimeType)").execute()
    # logging.debug(folder)
    folder = results.get('files', [])
    for item in folder:
        filepath = os.path.join(des, item['name'])
        if str(item['mimeType']) == str('application/vnd.google-apps.folder'):
            if not os.path.isdir(filepath):
                os.mkdir(filepath)
            listfolders(service, filepath)  # LOOP un-till the files are found
        elif not os.path.exists(filepath):
            print(filepath)
            downloadfiles(service, item['id'], item['name'], des)
    return folder


# To Download Files
def downloadfiles(service, dowid, name, dfilespath):
    request = service.files().get_media(fileId=dowid)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
    with io.open(dfilespath + "/" + name, 'wb') as f:
        fh.seek(0)
        f.write(fh.read())


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder-id', required=True)
    parser.add_argument('-O', '--output', dest='output')
    args = parser.parse_args()
    Folder_id: str = f"'{args.folder_id}'"
    bfolderpath: str = args.output
    if bfolderpath is None:
        bfolderpath = './Folder/'

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)  # credentials.json download from drive API
            creds = flow.run_local_server()
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    # Call the Drive v3 API

    results = service.files().list(q=Folder_id + " in parents",
                                   fields="nextPageToken, files(id, name, mimeType)").execute()
    items = results.get('files', [])
    if not items:
        print('No files found.')
    else:
        print('Files:')
        if not os.path.exists(bfolderpath):
            os.makedirs(bfolderpath)
        for item in items:
            filepath = os.path.join(bfolderpath, item['name'])
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                if not os.path.exists(filepath):
                    os.mkdir(filepath)
                listfolders(service, item['id'], filepath)
            elif not os.path.exists(filepath):
                print(filepath)
                downloadfiles(service, item['id'], item['name'], bfolderpath)


if __name__ == '__main__':
    main()
