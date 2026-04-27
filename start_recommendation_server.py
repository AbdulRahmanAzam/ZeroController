"""
Quick launcher for the recommendation server.
Run from the project root:  python start_recommendation_server.py
"""
import uvicorn

if __name__ == '__main__':
    uvicorn.run(
        'recommendation_system.server:app',
        host='0.0.0.0',
        port=8001,
        reload=False,
        log_level='info',
    )
