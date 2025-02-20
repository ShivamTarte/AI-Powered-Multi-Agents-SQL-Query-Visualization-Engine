from textwrap import dedent
import openai
import gradio as gr
from crewai import Agent, Task, Crew, LLM
from sqlalchemy import create_engine
from crewai.tools import tool
from langchain_community.tools import (
    ListSQLDatabaseTool,
    InfoSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities import SQLDatabase
import pymysql
import plotly.express as px
import pandas as pd
from langchain_experimental.utilities import PythonREPL
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import uuid
 
load_dotenv('agents.env')
 
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
gpt_key=os.getenv("gpt_key")
 
class SQLAgent:
    db = None
    llm = None
 
    def __init__(self, db_user: str, db_password: str, db_host: str, db_name: str):
        self.engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        SQLAgent.db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        client = openai.OpenAI(api_key=gpt_key)
        SQLAgent.llm = LLM(model="gpt-4o", api_key=gpt_key)        
 
        self.sql_agent = Agent(
            name="SQL Writer",
            role="Database Query Expert",
            goal="Generate optimized and accurate SQL queries and return results in Pandas DataFrame format.",
            backstory=dedent(
                """
                You are a highly skilled database engineer specializing in SQL queries.
                Your role is to extract structured data from databases and return it in a Pandas DataFrame format.
 
                - Use `list_tables` to find available tables.
                - Use `tables_schema` to understand table structures.
                - Use `check_sql` to verify query correctness.
                - Use `execute_sql` to run queries safely and return results as a Pandas DataFrame.
                """
            ),
            llm=SQLAgent.llm,
            tools=[
                SQLAgent.list_tables,
                SQLAgent.tables_schema,
                SQLAgent.check_sql,
                SQLAgent.execute_sql,
            ],
            allow_delegation=True,
            verbose=True,
            memory=False
        )
 
        self.visualization_agent = Agent(
            name="Visualization Expert",
            role="Data Visualization Specialist",
            goal="Create visualizations exclusively from Pandas DataFrame output provided by the SQL Writer.",
            backstory=dedent(
                """
                You are a data visualization specialist. Your job is to transform structured query results into insightful visualizations.
               
                - Work *only* with in-memory Pandas DataFrame provided by the SQL Writer.
                - Ensure clear, readable, and properly formatted visualizations.
               
                Visualization Guidelines:
                - **Matplotlib Table** (`plt.table`) for tabular data representation, ensuring proper alignment, border styling, and readable font size.
                - **Bar Chart** (Matplotlib/Plotly) for categorical comparisons.
                - **Line Chart** (Matplotlib/Plotly) for time-series trends.
                - **Scatter Plot** (Matplotlib/Plotly) for correlations.
               
                Restrictions:
                - **No File Operations**: Do *not* load, save, or request external files.
                - **No External API Calls**: Work with in-memory data only.
                - **Ensure proper column selection** for the appropriate chart type.
                """
            ),
            llm=SQLAgent.llm,
            tools=[SQLAgent.python_repl],
            allow_delegation=False,
            verbose=True,
            memory=False
        )
 
        self.extract_data = Task(
            description="Extract only the data required to answer the query: {query} and return it as a Pandas DataFrame.",
            expected_output="Pandas DataFrame containing the query result.",
            agent=self.sql_agent,
        )
 
        self.visualize_data = Task(
            description="Generate a visualization using the provided Pandas DataFrame.",
            expected_output="Plotly or Matplotlib visualization and insights about plot and 'don't' write path name of image or any code",
            agent=self.visualization_agent,
            context=[self.extract_data]
        )
 
        self.crew = Crew(
            agents=[self.sql_agent, self.visualization_agent],
            tasks=[self.extract_data, self.visualize_data],
            process="sequential",
            verbose=True,
            memory=False,
            output_log_file="crew.log",
        )
 
    @staticmethod
    @tool("list_tables")
    def list_tables() -> str:
        """List all available tables in the database."""
        return ListSQLDatabaseTool(db=SQLAgent.db).invoke("")
 
    @staticmethod
    @tool("tables_schema")
    def tables_schema(tables: str) -> str:
        """Retrieve schema and sample rows for given tables."""
        return InfoSQLDatabaseTool(db=SQLAgent.db).invoke(tables)
 
    @staticmethod
    @tool("execute_sql")
    def execute_sql(sql_query: str) -> str:
        """Execute a SQL query."""
        return QuerySQLDataBaseTool(db=SQLAgent.db).invoke(sql_query)
 
    @staticmethod
    @tool("check_sql")
    def check_sql(sql_query: str) -> str:
        """Validate a SQL query."""
        return QuerySQLCheckerTool(db=SQLAgent.db, llm=SQLAgent.llm).invoke({"query": sql_query})
 
    @staticmethod
    @tool("python_repl")
    def python_repl(python_code: str) -> str:
        """
        Executes the provided Python code in a restricted REPL environment.
 
        - If the code generates a plot, it must be saved as an image.
        - The image should be saved with the filename **'visualization.png'**.
        - After plotting graphs, ensure the following lines are added to save the image:
 
        ```python
        plt.savefig('visualization.png', format='png', bbox_inches='tight')
        plt.close()
        ```
 
        - Returns the file path **'visualization.png'** to be used for display.
        - Returns insights of plot data
 
        """
        return PythonREPL().run(python_code)
 
       
    def run_query(self, user_query: str):
        try:
            inputs = {"query": user_query}
            result = self.crew.kickoff(inputs=inputs)
            return result
        except Exception as e:
            return f"An error occurred: {e}"
       

def run_query_and_return_output(user_query: str, history: list):
    result = sql_agent_instance.run_query(user_query)  # Get SQL query result
    image_path = os.path.abspath("visualization.png")  # Path to visualization

    # Gradio expects either a string or a structured response.
    return {"text": str(result), "files": [image_path]}
 
sql_agent_instance = SQLAgent(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)
 
gr.ChatInterface(
    fn=run_query_and_return_output, 
    type="messages"
).launch()
 
 
 
 