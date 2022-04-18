import os
import pathlib
import json
import shutil

import hashlib

from duplicates_engine import DuplicatesEngine

import app_env
from data_loader import DataLoader


class IndexStore:
    def __init__(self):
        self.engine = DuplicatesEngine()

    def dump_for_ideas(self, idea_id, with_own_project=True, project_ids=None, skip_exists=True):
        if project_ids is None:
            project_ids = []

        if skip_exists:
            if len(project_ids) !=0:
                project_ids = list(filter(lambda x: self.__is_file_exists(idea_id, project_id=x) is False, project_ids))

            if with_own_project is True:
                if self.__is_file_exists(idea_id):
                    with_own_project = False

        if with_own_project is False and len(project_ids) == 0:
            print("[IndexStore] Skip build index")
            return
            
        result = self.engine.get_idea_duplicates(idea_id, with_own_project=with_own_project, project_ids=project_ids)

        self.__delete_index_dir(idea_id)
        self.__touch_idea_id_file(idea_id)
         
        for key in result:
            data = result[key]
            path = self.index_dir_path(idea_id, key)
            print("[IndexStore] Dump idea {idea_id} for {project} to file {file_path}".
                  format(idea_id=idea_id, project=key, file_path=path))
            self.__save_to_file(path, data)

    def dump_for_project(self, project_id, with_own_project=True, project_ids=None, skip_exists=True):
        idea_ids = DataLoader().ideas_in_project(project_id)

        for idea_id in idea_ids:
            self.dump_for_ideas(idea_id, with_own_project=with_own_project,
                                project_ids=project_ids, skip_exists=skip_exists)

    def index_dir_path(self, idea_id=None, project_id=None):
        parts = ['idea_indexes']
        if idea_id is not None:
            parts += self.idea_id_to_parts(idea_id)

            if project_id is not None:
                parts += [str(project_id) + ".json"]
            else:
                parts += ["own_project.json"]

        name = os.path.join(*parts)

        return app_env.data_model_runtime_path(name, not_builtin=True)

    @staticmethod
    def idea_id_to_parts(idea_id):
        h = hashlib.md5(str(idea_id).encode('utf-8'))

        d = h.hexdigest()

        return [d[0:2], d[2:4], d[4:]]

    @staticmethod
    def __create_index_dir(path):
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        
    def __delete_index_dir(self, idea_id):
        path = self.index_dir_path(idea_id)
        print("[IndexStore] Remove dir {dir}".
                  format(dir=os.path.dirname(path)))
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def __save_to_file(self, path, json_data):
        self.__create_index_dir(path)
        with open(path, 'w') as json_file:
            json.dump(json_data, json_file)

    def __touch_idea_id_file(self, idea_id):
        path = self.index_dir_path(idea_id)
        dir_name = os.path.dirname(path)
        path = os.path.join(dir_name, str(idea_id))
        if os.path.exists(path):
            return
        else:
            self.__create_index_dir(path)
            with open(path, 'w') as json_file:
                json_file.write(str(idea_id))

    def __is_file_exists(self, idea_id, project_id=None):
        return os.path.exists(self.index_dir_path(idea_id=idea_id, project_id=project_id))
