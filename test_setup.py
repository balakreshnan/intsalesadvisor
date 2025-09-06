#!/usr/bin/env python3
"""
Test script to verify the Voice Live Web Server setup
"""

import os
from dotenv import load_dotenv

def test_environment():
    """Test if all required environment variables are set"""
    load_dotenv()
    
    required_vars = [
        'AZURE_VOICE_LIVE_ENDPOINT',
        'AI_FOUNDRY_AGENT_ID', 
        'AI_FOUNDRY_PROJECT_NAME',
        'AZURE_VOICE_LIVE_API_VERSION'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.environ.get(var)
        if not value or value.startswith('<'):
            missing_vars.append(var)
        else:
            print(f"✅ {var}: {value[:50]}...")
    
    if missing_vars:
        print(f"\n❌ Missing required environment variables: {missing_vars}")
        return False
    else:
        print(f"\n✅ All required environment variables are set!")
        return True

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import flask
        import flask_socketio
        import azure.identity
        import sounddevice
        import numpy
        import websocket
        print("✅ All required modules imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Voice Live Web Server Setup...")
    print("=" * 50)
    
    env_ok = test_environment()
    imports_ok = test_imports()
    
    if env_ok and imports_ok:
        print("\n🎉 Setup test passed! You can run the voice live web server.")
        print("\nTo start the server, run:")
        print("python voice_live_web_server.py")
        print("\nThen open your browser to: http://localhost:5000")
    else:
        print("\n❌ Setup test failed. Please fix the issues above.")
