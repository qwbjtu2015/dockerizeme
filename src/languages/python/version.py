#!/usr/bin/env python

from __future__ import with_statement
from neo4j import GraphDatabase
import getopt
import sys
import json

def convert_ver_number(ver):
    if '*' in ver:
        ver.replace('.*', '')
    cnt = len(ver.split('.'))
    if cnt == 2:
        return 10*int(ver.replace('.',''))
    else:
        return int(ver.replace('.',''))

def get_version(versions):
    if not versions:
        return "2.7"
    min = 2.7
    max = 3.8
    target = None
    for v in versions:
        v = v.replace(' ', '')
        if "==" in v:
            v = v.replace("==", '')
            target = convert_ver_number(v)
            break
        elif ">=" in v:
            v = v.replace(">=", '')
            min = convert_ver_number(v)
        elif ">" in v:
            v = v.replace(">", '')
            min = convert_ver_number(v)+1
    if target:
        v = str(target).replace('*', '')
    else:
        v = int(min/10)
        if v > 38:
            v = 38
    return ".".join(list(str(v)))

def get_image_version(pkgs):
    uri = "bolt://60.245.211.161:7687"
    pkgs = pkgs.strip().split(',')
    #uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))
    versions = []
    with driver.session() as session:
        records = session.run("MATCH (p:package)-[:pkg2ver]->(v:version) WHERE p.name in {imports} RETURN v", imports=pkgs)
        for record in records:
            version = record["v"].get("require_python")
            if version:
                versions.extend(version.split(";"))
        return get_version(versions)

def main():
    """
    Usage
    -----
    python version.py <pkgs>
    """

    opts, args = getopt.getopt(sys.argv[1:], '', [])
    if not args:
        raise Exception('Usage: python parse.py <pkgs>')

    data = {}
    try:
        data["imageVersion"] = get_image_version(args[0])
    except:
        data["imageVersion"] = "2.7.13"
    print(json.dumps(data))


if __name__ == '__main__':
    main()
