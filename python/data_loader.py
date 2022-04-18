import pandas as pd

import app_env

import sqlalchemy

from search_engine_client import SearchEngineClient


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def is_search_engine():
        return app_env.ml_params_scoring_search_filter() == 'search_engine'

    @staticmethod
    def is_none():
        return app_env.ml_params_scoring_search_filter() == 'full_db'

    def ideas(self, idea_id, with_own_project=True, project_ids=None):
        if self.is_search_engine():
            return self.ideas_from_search_engine(idea_id, project_ids=project_ids)
        elif self.is_none():
            return self.ideas_from_db(idea_id, with_own_project=with_own_project, project_ids=project_ids)

    def ideas_from_db(self, idea_id, with_own_project=True, project_ids=None):
        if project_ids is None:
            project_ids = []
        sql, params = self.get_ideas_sql_with_params(idea_id, with_own_project=with_own_project, project_ids=project_ids)

        return pd.read_sql_query(sql, app_env.db_engine(), params=params)

    def ideas_from_search_engine(self, idea_id, project_ids=None):
        client = SearchEngineClient()

        content = self.get_idea_content(idea_id)

        if content is None:
            return pd.DataFrame()

        idea_ids = client.get_ideas(content, limit=21)

        if len(idea_ids) == 0:
            return pd.DataFrame()

        sql,params = self.get_ideas_sql_with_params(idea_id, with_own_project=False, other_idea_ids=idea_ids)

        return pd.read_sql_query(sql, app_env.db_engine(), params=params)

    @staticmethod
    def get_ideas_sql_with_params(idea_id, with_own_project=True, project_ids=None, other_idea_ids=None):
        if project_ids is None:
            project_ids = []
        if other_idea_ids is None:
            other_idea_ids = []
        params = {"idea_id": idea_id}

        inner_sql = """
SELECT id, content, parent_id
FROM pages as pages_for_dups
WHERE {project_conditions} 
      id != pages.id
      AND content IS NOT NULL
      AND type IN ('Pim::Idea', 'Pim::VndIdea', 'Pim::BestPracticeIdea')
        """

        project_conditions = []
        if with_own_project:
            project_conditions.append("parent_id = pages.parent_id")

        if len(project_ids) != 0:
            project_conditions.append("parent_id=ANY(ARRAY[:project_ids])")
            params["project_ids"] = project_ids

        if len(other_idea_ids) != 0:
            project_conditions.append("id=ANY(ARRAY[:other_idea_ids])")
            params["other_idea_ids"] = other_idea_ids

        if with_own_project is False and len(project_ids) == 0 and len(other_idea_ids) == 0:
            project_conditions.append("1=0")

        project_conditions = " OR ".join(project_conditions)
        if len(project_conditions) != 0:
            project_conditions += " AND "

        inner_sql = inner_sql.format(project_conditions=project_conditions)

        sql = """
SELECT pages.id, pages.id AS id1,  pages_for_dups.id AS id2, 
       pages.content, pages.content AS content1, pages_for_dups.content AS content2,
       project1.real_project_id as project_id1, project2.real_project_id AS project_id2,
       (project1.real_project_id = project2.real_project_id) AS is_own_project,
       '{scoring_search_filter}' AS scoring_search_filter       
FROM pages
    LEFT OUTER JOIN LATERAL({inner_sql}) pages_for_dups ON pages.id != pages_for_dups.id
    LEFT OUTER JOIN LATERAL(SELECT COALESCE(project_pages.parent_id, project_pages.id) AS real_project_id, 
                                   project_pages.id AS project_id 
                            FROM pages AS project_pages 
                            WHERE project_pages.id = pages.parent_id) project1 
                    ON pages.parent_id = project1.project_id
    LEFT OUTER JOIN LATERAL(SELECT COALESCE(project_pages.parent_id, project_pages.id) AS real_project_id, 
                                   project_pages.id AS project_id 
                            FROM pages AS project_pages 
                            WHERE project_pages.id = pages_for_dups.parent_id) project2 
                    ON pages_for_dups.parent_id = project2.project_id
WHERE pages.id = :idea_id
    AND pages.type IN ('Pim::Idea', 'Pim::VndIdea', 'Pim::BestPracticeIdea')
        """

        sql = sql.format(inner_sql=inner_sql,
                         scoring_search_filter=app_env.ml_params_scoring_search_filter())

        sql = sqlalchemy.text(sql)

        return sql, params

    def ideas_in_project(self, project_id):
        sql = """
SELECT id
FROM pages
WHERE CASE
        WHEN (select type from pages where id = :project_id) = 'Pim::ParentProject'
            THEN parent_id in (SELECT id FROM pages WHERE parent_id = :project_id AND type ILIKE 'Pim::%Project')
        ELSE
          parent_id = :project_id
      END
      AND content IS NOT NULL
      AND type IN ('Pim::Idea', 'Pim::VndIdea', 'Pim::BestPracticeIdea')
        """

        sql = sqlalchemy.text(sql)
        df = pd.read_sql_query(sql, app_env.db_engine(), params={'project_id': project_id})

        return list(df['id'])

    def get_idea_content(self, idea_id):
        sql = """
SELECT content
FROM pages
WHERE id = :idea_id
    AND content IS NOT NULL
    AND type IN ('Pim::Idea', 'Pim::VndIdea', 'Pim::BestPracticeIdea')   
        """

        sql = sqlalchemy.text(sql)
        df = pd.read_sql_query(sql, app_env.db_engine(), params={'idea_id': idea_id})

        if len(df) is None:
            return None
        else:
            return df['content'].values[0]
