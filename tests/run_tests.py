#!/usr/bin/env python3
"""
Simple test runner script for the churn prediction project.
"""

import subprocess
import sys

def run_tests():
    """Run the test suite."""
    try:
        subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/', 
            '-v', 
            '--cov=scripts', 
            '--cov-report=term-missing',
            '--cov-report=html'
        ], check=True)
        
        print("\n All tests passed")
        print("📊 Coverage report generated in htmlcov/index.html")
        
    except subprocess.CalledProcessError as e:
        print(f"\n Tests failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests() 