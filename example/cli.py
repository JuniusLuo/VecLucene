
import argparse
import requests
import sys

def upload_file(url: str, file_path: str):
    with open(file_path, 'rb') as f:
        resp = requests.post(url=url, files={'file': (f.name, f, "text/plain")})
        print(resp.json())


def upload_file_with_fields(url: str, file_path: str):
    with open(file_path, 'rb') as f:
        field1 = '{"name": "field1", "string_value": "str1"}'
        field2 = '{"name": "field2", "numeric_value": 2}'
        doc_fields = '{"fields": ' + f'[{field1}, {field2}]' + '}'
        fields = {"fields": f'{doc_fields}'}
        resp = requests.post(url=url, files={'file': (f.name, f, "text/plain")}, data=fields) 
        print(resp.json())


def commit(url: str):
    resp = requests.post(url=url)
    print(resp.json())


def query(url: str, query_string: str):
    query_request = '{"query": ' + f'"{query_string}"' + '}'
    resp = requests.get(url=url, data=query_request)
    print(resp.json())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--file", type=str)
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    match args.op:
        case "upload":
            if args.file is None:
                print("please input the text file path")
            url += "/add_doc"
            upload_file(url, args.file)

        case "commit":
            url += "/commit"
            commit(url)

        case "query":
            if args.query is None:
                print("please input the query string")
            url += "/query"
            query(url, args.query)

        case _:
            print("supported op: upload, commit, query")



#url = 'http://127.0.0.1:8000/add_doc'
#upload_file(url, "tests/testfiles/single_sentence.txt")
#upload_file_with_fields(url, "tests/testfiles/single_sentence.txt")

#url = 'http://127.0.0.1:8000/commit'
#commit(url)

#url = 'http://127.0.0.1:8000/query'
#query(url, "A person is eating food")

