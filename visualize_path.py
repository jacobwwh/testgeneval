#print path for bug detection task
import json

bugdetect_path = "bugdetect-data/test-filtered-paths-raw.jsonl"

with open(bugdetect_path, "r") as f:
    flines = f.readlines()
    data = json.loads(flines[0])
    paths = data['paths']
    print(data['problem_id'], data['submission_id'])
    print(f'number of paths: {len(paths)}')
    #for path in paths:
        #print(len(path))

    path = paths[10]
    with open('tmp-path', 'w') as fw:
        for block in path:
            fw.write(f'{block}\n')
        

    '''for i, line in enumerate(f):
        data = json.loads(line)
        paths = data['paths']
        print(f'number of paths: {len(paths)}')
        for path in paths:
            print(len(path))
        break'''