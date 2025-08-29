"""
API Integration Examples
Blog 4: Connecting AI to External Data - Section 2

Demonstrates how to connect AI agents to external APIs and services,
enabling them to fetch real-time data and interact with web services.

Use cases: Web APIs, databases, file systems, real-time data feeds
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.llm_providers.base_llm import get_default_provider, LLMResponse
from typing import Dict, List, Any, Optional
import json
import requests
import time
from datetime import datetime
import sqlite3
import csv
from pathlib import Path


class APIConnector:
    """
    Handles connections to various external APIs and data sources.
    Provides a unified interface for AI agents to access external data.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Agent-Demo/1.0'
        })
    
    def fetch_json_api(self, url: str, params: Optional[Dict] = None, 
                      headers: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fetch data from a JSON API.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Dictionary containing API response
        """
        try:
            if headers:
                self.session.headers.update(headers)
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return {
                'success': True,
                'data': response.json(),
                'status_code': response.status_code,
                'headers': dict(response.headers)
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def post_json_api(self, url: str, data: Dict[str, Any], 
                     headers: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Send data to a JSON API via POST.
        
        Args:
            url: API endpoint URL
            data: Data to send
            headers: Additional headers
            
        Returns:
            Dictionary containing API response
        """
        try:
            if headers:
                self.session.headers.update(headers)
            
            response = self.session.post(url, json=data, timeout=10)
            response.raise_for_status()
            
            return {
                'success': True,
                'data': response.json() if response.content else None,
                'status_code': response.status_code
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def fetch_public_api(self, api_name: str, endpoint: str, 
                        params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fetch data from common public APIs.
        
        Args:
            api_name: Name of the API service
            endpoint: Specific endpoint
            params: Query parameters
            
        Returns:
            Formatted API response
        """
        api_bases = {
            'jsonplaceholder': 'https://jsonplaceholder.typicode.com',
            'httpbin': 'https://httpbin.org',
            'reqres': 'https://reqres.in/api',
            'cat_facts': 'https://catfact.ninja',
            'dog_api': 'https://dog.ceo/api'
        }
        
        base_url = api_bases.get(api_name)
        if not base_url:
            return {
                'success': False,
                'error': f"Unknown API: {api_name}",
                'available_apis': list(api_bases.keys())
            }
        
        url = f"{base_url}/{endpoint}"
        return self.fetch_json_api(url, params)


class DatabaseConnector:
    """
    Handles connections to databases (SQLite for demo purposes).
    In production, you'd support PostgreSQL, MySQL, MongoDB, etc.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = None
        self._setup_demo_database()
    
    def _setup_demo_database(self):
        """Create demo database with sample data."""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row  # Enable column access by name
        
        cursor = self.connection.cursor()
        
        # Create sample tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary INTEGER,
                hire_date TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                product TEXT NOT NULL,
                amount DECIMAL(10,2),
                sale_date TEXT,
                employee_id INTEGER,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
        """)
        
        # Insert sample data
        employees = [
            ("Alice Johnson", "Engineering", 95000, "2022-01-15"),
            ("Bob Smith", "Sales", 75000, "2021-06-10"),
            ("Carol Davis", "Marketing", 70000, "2023-03-20"),
            ("David Wilson", "Engineering", 88000, "2022-09-05"),
            ("Eva Brown", "Sales", 82000, "2021-11-12")
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO employees (name, department, salary, hire_date) VALUES (?, ?, ?, ?)",
            employees
        )
        
        sales = [
            ("Software License", 5000.00, "2024-01-15", 2),
            ("Consulting Service", 12000.00, "2024-01-20", 5),
            ("Software License", 3000.00, "2024-02-01", 2),
            ("Training Package", 8000.00, "2024-02-10", 5),
            ("Custom Development", 25000.00, "2024-02-15", 1)
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO sales (product, amount, sale_date, employee_id) VALUES (?, ?, ?, ?)",
            sales
        )
        
        self.connection.commit()
    
    def query(self, sql: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            Query results as dictionary
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, params or ())
            
            if sql.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                return {
                    'success': True,
                    'data': [dict(row) for row in rows],
                    'row_count': len(rows)
                }
            else:
                self.connection.commit()
                return {
                    'success': True,
                    'rows_affected': cursor.rowcount
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table structure."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            return {
                'success': True,
                'table': table_name,
                'columns': [dict(col) for col in columns]
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


class FileSystemConnector:
    """
    Handles file system operations for AI agents.
    Provides safe file access with proper error handling.
    """
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self._setup_demo_files()
    
    def _setup_demo_files(self):
        """Create demo files for testing."""
        # Create sample CSV file
        csv_data = [
            ["Name", "Age", "City", "Occupation"],
            ["Alice", "28", "San Francisco", "Engineer"],
            ["Bob", "34", "New York", "Designer"],
            ["Carol", "31", "Chicago", "Analyst"],
            ["David", "29", "Seattle", "Developer"]
        ]
        
        csv_file = self.base_path / "sample_data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        
        # Create sample JSON file
        json_data = {
            "products": [
                {"id": 1, "name": "Laptop", "price": 999.99, "category": "Electronics"},
                {"id": 2, "name": "Book", "price": 19.99, "category": "Education"},
                {"id": 3, "name": "Coffee", "price": 4.99, "category": "Food"}
            ],
            "metadata": {
                "created": "2024-01-01",
                "version": "1.0"
            }
        }
        
        json_file = self.base_path / "sample_data.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Create sample text file
        text_file = self.base_path / "notes.txt"
        with open(text_file, 'w') as f:
            f.write("This is a sample text file.\nIt contains multiple lines.\nUseful for testing file operations.")
    
    def read_file(self, filename: str) -> Dict[str, Any]:
        """Read file contents safely."""
        try:
            file_path = self.base_path / filename
            
            if not file_path.exists():
                return {'success': False, 'error': f"File {filename} not found"}
            
            if not file_path.is_file():
                return {'success': False, 'error': f"{filename} is not a file"}
            
            # Determine file type and read accordingly
            if filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return {'success': True, 'data': data, 'type': 'json'}
            
            elif filename.endswith('.csv'):
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                return {'success': True, 'data': data, 'type': 'csv'}
            
            else:
                with open(file_path, 'r') as f:
                    data = f.read()
                return {'success': True, 'data': data, 'type': 'text'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def write_file(self, filename: str, data: Any, file_type: str = 'text') -> Dict[str, Any]:
        """Write data to file safely."""
        try:
            file_path = self.base_path / filename
            
            if file_type == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif file_type == 'csv' and isinstance(data, list):
                with open(file_path, 'w', newline='') as f:
                    if data and isinstance(data[0], dict):
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    else:
                        writer = csv.writer(f)
                        writer.writerows(data)
            else:
                with open(file_path, 'w') as f:
                    f.write(str(data))
            
            return {'success': True, 'message': f"File {filename} written successfully"}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def list_files(self) -> Dict[str, Any]:
        """List files in the base directory."""
        try:
            files = []
            for item in self.base_path.iterdir():
                if item.is_file():
                    files.append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
            
            return {'success': True, 'files': files}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class DataConnectedAgent:
    """
    AI agent with access to external data sources.
    Can query APIs, databases, and files to answer questions.
    """
    
    def __init__(self):
        self.provider = get_default_provider()
        self.api_connector = APIConnector()
        self.db_connector = DatabaseConnector()
        self.file_connector = FileSystemConnector()
    
    def answer_with_data(self, question: str, data_sources: List[str]) -> str:
        """
        Answer a question using specified data sources.
        
        Args:
            question: User's question
            data_sources: List of data sources to use
            
        Returns:
            AI-generated answer based on data
        """
        context_data = []
        
        for source in data_sources:
            if source.startswith('api:'):
                # API data source
                api_info = source.split(':', 2)
                if len(api_info) == 3:
                    _, api_name, endpoint = api_info
                    result = self.api_connector.fetch_public_api(api_name, endpoint)
                    if result['success']:
                        context_data.append(f"API data from {api_name}/{endpoint}: {json.dumps(result['data'])}")
                    else:
                        context_data.append(f"API error: {result['error']}")
            
            elif source.startswith('db:'):
                # Database query
                query = source.split(':', 1)[1]
                result = self.db_connector.query(query)
                if result['success']:
                    context_data.append(f"Database result: {json.dumps(result['data'])}")
                else:
                    context_data.append(f"Database error: {result['error']}")
            
            elif source.startswith('file:'):
                # File data source
                filename = source.split(':', 1)[1]
                result = self.file_connector.read_file(filename)
                if result['success']:
                    context_data.append(f"File content ({filename}): {json.dumps(result['data']) if result['type'] != 'text' else result['data']}")
                else:
                    context_data.append(f"File error: {result['error']}")
        
        # Generate answer using context data
        context = "\n\n".join(context_data) if context_data else "No data available"
        
        prompt = f"""Answer the following question using the provided data context:

Question: {question}

Data Context:
{context}

Provide a comprehensive answer based on the available data:"""
        
        response = self.provider.generate(prompt)
        return response.content
    
    def analyze_data_trends(self, data_source: str) -> str:
        """Analyze trends in data from a specific source."""
        if data_source.startswith('db:'):
            query = data_source.split(':', 1)[1]
            result = self.db_connector.query(query)
            
            if result['success'] and result['data']:
                data_json = json.dumps(result['data'], indent=2)
                
                prompt = f"""Analyze the following data and identify key trends, patterns, and insights:

Data:
{data_json}

Provide analysis including:
1. Key trends and patterns
2. Notable statistics
3. Actionable insights
4. Recommendations

Analysis:"""
                
                response = self.provider.generate(prompt)
                return response.content
            else:
                return f"Error accessing data: {result.get('error', 'Unknown error')}"
        
        return "Unsupported data source for trend analysis"
    
    def close_connections(self):
        """Clean up all connections."""
        self.db_connector.close()


def demo_data_connections():
    """Comprehensive demonstration of AI data connection capabilities."""
    print("=== AI Data Connections Demo ===\n")
    
    try:
        agent = DataConnectedAgent()
        print(f"🤖 Using provider: {agent.provider.provider_name}")
        print("🔗 Data Connections: API, Database, File System")
        print("-" * 60 + "\n")
        
        # 1. API Connection Demo
        print("1. API Connection Demo")
        print("Fetching data from public APIs:")
        
        # Test JSONPlaceholder API
        api_result = agent.api_connector.fetch_public_api('jsonplaceholder', 'users/1')
        if api_result['success']:
            user_data = api_result['data']
            print(f"✅ User data from API: {user_data['name']} ({user_data['email']})")
        else:
            print(f"❌ API error: {api_result['error']}")
        
        # Test with posts
        posts_result = agent.api_connector.fetch_public_api('jsonplaceholder', 'posts', {'userId': 1})
        if posts_result['success']:
            posts = posts_result['data']
            print(f"✅ Retrieved {len(posts)} posts from user")
        
        print()
        
        # 2. Database Connection Demo
        print("2. Database Connection Demo")
        print("Querying SQLite database with sample data:")
        
        # Query employees
        employees = agent.db_connector.query("SELECT name, department, salary FROM employees ORDER BY salary DESC")
        if employees['success']:
            print("✅ Employee data:")
            for emp in employees['data'][:3]:
                print(f"   {emp['name']} - {emp['department']} - ${emp['salary']:,}")
        
        # Query sales with join
        sales_query = """
            SELECT e.name, s.product, s.amount, s.sale_date 
            FROM sales s 
            JOIN employees e ON s.employee_id = e.id 
            ORDER BY s.amount DESC
        """
        sales = agent.db_connector.query(sales_query)
        if sales['success']:
            print("✅ Top sales:")
            for sale in sales['data'][:3]:
                print(f"   {sale['name']}: {sale['product']} - ${sale['amount']:,.2f}")
        
        print()
        
        # 3. File System Connection Demo
        print("3. File System Connection Demo")
        print("Reading various file formats:")
        
        # List available files
        files = agent.file_connector.list_files()
        if files['success']:
            print("✅ Available files:")
            for file_info in files['files']:
                print(f"   {file_info['name']} ({file_info['size']} bytes)")
        
        # Read CSV file
        csv_data = agent.file_connector.read_file('sample_data.csv')
        if csv_data['success']:
            print(f"✅ CSV data: {len(csv_data['data'])} records loaded")
            print(f"   Sample: {csv_data['data'][0]}")
        
        # Read JSON file
        json_data = agent.file_connector.read_file('sample_data.json')
        if json_data['success']:
            print(f"✅ JSON data: {len(json_data['data']['products'])} products loaded")
        
        print()
        
        # 4. AI Question Answering with Data
        print("4. AI Question Answering with External Data")
        print("AI agent answers questions using connected data sources:")
        
        questions_and_sources = [
            (
                "Who are the top 3 highest paid employees?",
                ["db:SELECT name, department, salary FROM employees ORDER BY salary DESC LIMIT 3"]
            ),
            (
                "What products do we have in our inventory and what are their prices?",
                ["file:sample_data.json"]
            ),
            (
                "Show me the sales performance by employee.",
                ["db:SELECT e.name, COUNT(*) as sales_count, SUM(s.amount) as total_sales FROM sales s JOIN employees e ON s.employee_id = e.id GROUP BY e.name ORDER BY total_sales DESC"]
            )
        ]
        
        for question, sources in questions_and_sources:
            print(f"\nQ: {question}")
            answer = agent.answer_with_data(question, sources)
            print(f"A: {answer}")
        
        print()
        
        # 5. Data Trend Analysis
        print("5. AI Data Trend Analysis")
        print("AI analyzes patterns in business data:")
        
        analysis_query = """
            SELECT 
                strftime('%Y-%m', sale_date) as month,
                COUNT(*) as transaction_count,
                SUM(amount) as total_revenue,
                AVG(amount) as avg_transaction
            FROM sales 
            GROUP BY strftime('%Y-%m', sale_date)
            ORDER BY month
        """
        
        analysis = agent.analyze_data_trends(f"db:{analysis_query}")
        print("Sales trend analysis:")
        print(analysis[:400] + "..." if len(analysis) > 400 else analysis)
        
        print()
        
        # 6. Real-time API Integration
        print("6. Real-time Data Integration")
        print("Combining multiple data sources for comprehensive answers:")
        
        complex_question = "Based on our employee and sales data, who should we consider for promotion?"
        complex_sources = [
            "db:SELECT e.name, e.department, COUNT(*) as sales_count, SUM(s.amount) as total_sales FROM sales s JOIN employees e ON s.employee_id = e.id GROUP BY e.name",
            "db:SELECT name, hire_date FROM employees ORDER BY hire_date"
        ]
        
        complex_answer = agent.answer_with_data(complex_question, complex_sources)
        print(f"Q: {complex_question}")
        print(f"A: {complex_answer}")
        
        print("\n" + "=" * 70)
        print("🎯 Data Connections Key Insights:")
        print("   ✅ AI can access real-time external data")
        print("   ✅ Multiple data sources can be combined")
        print("   ✅ Structured queries enable precise data retrieval")
        print("   ✅ File formats (CSV, JSON, text) are all supported")
        print("   ⚠️  Always validate and sanitize data inputs")
        print("   ⚠️  Implement proper error handling for network calls")
        print("   💡 Cache frequently accessed data for performance")
        print("   💡 Use database indexes for complex queries")
        
        # Cleanup
        agent.close_connections()
        
    except Exception as e:
        print(f"❌ Error during data connections demo: {e}")
        print("💡 Check your internet connection and data sources")


def demo_api_integration():
    """Focused demo on API integration patterns."""
    print("\n=== API Integration Patterns Demo ===\n")
    
    connector = APIConnector()
    
    # Test different public APIs
    apis_to_test = [
        ('jsonplaceholder', 'posts/1', "Sample blog post data"),
        ('cat_facts', 'fact', "Random cat fact"),
        ('dog_api', 'breeds/image/random', "Random dog image"),
    ]
    
    for api_name, endpoint, description in apis_to_test:
        print(f"Testing {api_name} - {description}")
        result = connector.fetch_public_api(api_name, endpoint)
        
        if result['success']:
            print(f"✅ Success: {str(result['data'])[:100]}...")
        else:
            print(f"❌ Failed: {result['error']}")
        print()
    
    print("💡 API Integration Best Practices:")
    print("   • Always handle rate limits and timeouts")
    print("   • Implement retry logic with exponential backoff")
    print("   • Cache responses when appropriate")
    print("   • Validate API responses before using")
    print("   • Use environment variables for API keys")


if __name__ == "__main__":
    demo_data_connections()
    demo_api_integration()