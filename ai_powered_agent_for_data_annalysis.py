from textwrap import dedent
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

DB_USER = "root"
DB_PASSWORD = "Arcsaber0001"
DB_HOST = "localhost"
DB_NAME = "world"

class SQLAgent:
    db = None
    llm = None

    def __init__(self, db_user: str, db_password: str, db_host: str, db_name: str):
        self.engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        SQLAgent.db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        SQLAgent.llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

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
                - **Styled DataFrame** (`Pandas .style`) for tabular data.
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
            expected_output="Plotly or Matplotlib visualization.",
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
        Execute the provided Python code in a restricted REPL environment.
        Used primarily for data visualization.
        """
        return PythonREPL().run(python_code)

    def run_query(self, user_query: str):
        try:
            inputs = {"query": user_query}
            result = self.crew.kickoff(inputs=inputs)
            return result
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    sql_agent_instance = SQLAgent(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)

    print("\nSQL Query & Visualization Generator (CLI Version)")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your SQL Query in Natural Language: ").strip()
        
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        result = sql_agent_instance.run_query(user_query)
        print("\nQuery Result:\n", result)

# calculate and categorize surface area as per continents
# show me in tabular format population of each city in aghanistan
# show me countries according to continent in tabular format
# categorize top 3 continents with highest population
# show me in tabular format top 3 rows of country table