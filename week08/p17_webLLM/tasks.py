import time
import random
from celery import shared_task
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Task  # Import the Task model

# Database setup for task updates
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/taskdb")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@shared_task(bind=True)
def add_numbers(self, number):
    """
    A simple task that just waits for a random time between 15-20 seconds,
    updates status, and then marks as done.
    """
    task_id = self.request.id
    
    # Get a database session
    db = SessionLocal()
    
    try:
        print(f"Task {task_id} STARTED for number {number}")
        
        # Create or update task in database
        db_task = db.query(Task).filter(Task.id == task_id).first()
        if not db_task:
            db_task = Task(id=task_id, status="PROCESSING", result={"progress": 0})
            db.add(db_task)
        else:
            db_task.status = "PROCESSING"
            db_task.result = {"progress": 0}
        db.commit()
        
        # Generate a random sleep duration between 15-20 seconds
        sleep_duration = random.uniform(15.0, 20.0)
        print(f"Task {task_id} will process for {sleep_duration:.2f} seconds")
        
        # Calculate total steps (we'll divide the sleep into 10 steps)
        total_steps = 10
        sleep_per_step = sleep_duration / total_steps
        
        # Process in steps with progress updates
        for step in range(1, total_steps + 1):
            # Sleep for a portion of the total time
            time.sleep(sleep_per_step)
            
            # Calculate progress percentage
            progress = int((step / total_steps) * 100)
            
            # Update progress in database
            if db_task:
                db_task.result = {"progress": progress}
                db.commit()
            
            # Also update Celery task meta
            self.update_state(
                state="PROGRESS",
                meta={"progress": progress}
            )
            
            print(f"Task {task_id} progress: {progress}%")
        
        # Calculate the sum
        result = sum(range(number + 1))
        
        task_result = {
            "number": number,
            "result": result,
            "progress": 100,
            "processing_time": f"{sleep_duration:.2f} seconds",
            "message": f"Successfully calculated sum of numbers from 0 to {number}"
        }
        
        # Update task with result in database
        if db_task:
            db_task.status = "DONE"  # Using DONE instead of SUCCESS
            db_task.result = task_result
            db.commit()
            
        return task_result
    
    except Exception as e:
        print(f"Task {task_id} FAILED with error: {str(e)}")
        # Update task with error in database
        if db_task:
            db_task.status = "FAILED"
            db_task.result = {"error": str(e), "progress": 0}
            db.commit()
        raise
    
    finally:
        db.close()