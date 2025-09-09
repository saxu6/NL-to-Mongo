# Deployment Guide

## Production Deployment

### Prerequisites

- Python 3.8+
- MongoDB Atlas cluster
- OpenAI API key
- Server with at least 2GB RAM
- Domain name (optional)

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd nl_to_mongo_new

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env with production values
nano .env
```

**Production .env example:**
```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
ATLAS_CLUSTER_NAME=production-cluster
ATLAS_DATABASE_NAME=production_db
ATLAS_COLLECTION_NAME=main_collection

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-your-production-key
OPENAI_MODEL=gpt-4o-mini

# Application Configuration
LOG_LEVEL=INFO
MAX_RETRIES=3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "scripts/start_server.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  mongodb-query-translator:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

**Deploy with Docker:**
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 4. Cloud Deployment

#### AWS EC2

```bash
# Launch EC2 instance (t3.medium or larger)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Clone and deploy
git clone <repository-url>
cd nl_to_mongo_new
docker-compose up -d
```

#### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/mongodb-query-translator

# Deploy to Cloud Run
gcloud run deploy mongodb-query-translator \
  --image gcr.io/PROJECT-ID/mongodb-query-translator \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MONGODB_URI=$MONGODB_URI,OPENAI_API_KEY=$OPENAI_API_KEY
```

#### Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: python scripts/start_server.py" > Procfile

# Deploy
heroku create your-app-name
heroku config:set MONGODB_URI=$MONGODB_URI
heroku config:set OPENAI_API_KEY=$OPENAI_API_KEY
git push heroku main
```

### 5. Nginx Reverse Proxy

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 6. SSL Certificate

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 7. Monitoring

#### Health Checks

```bash
# Add to crontab for monitoring
*/5 * * * * curl -f http://localhost:8000/health || echo "Service down" | mail -s "Alert" admin@yourdomain.com
```

#### Log Management

```bash
# Install logrotate
sudo apt install logrotate

# Create logrotate config
sudo nano /etc/logrotate.d/mongodb-query-translator
```

**logrotate config:**
```
/var/log/mongodb-query-translator/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload mongodb-query-translator
    endscript
}
```

### 8. Performance Optimization

#### Database Indexes

```javascript
// Create indexes for common queries
db.users.createIndex({ "status": 1 })
db.users.createIndex({ "email": 1 })
db.orders.createIndex({ "user_id": 1, "created_at": -1 })
db.orders.createIndex({ "status": 1, "created_at": -1 })
```

#### Caching

```python
# Add Redis caching (optional)
pip install redis

# In your application
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

### 9. Security

#### API Authentication

```python
# Add API key authentication
from fastapi import HTTPException, Depends, Header

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != "your-secret-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Use in endpoints
@app.post("/query/generate")
async def generate_query(query: QueryRequest, api_key: str = Depends(verify_api_key)):
    # Your code here
```

#### Rate Limiting

```python
# Add rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Use in endpoints
@app.post("/query/generate")
@limiter.limit("10/minute")
async def generate_query(request: Request, query: QueryRequest):
    # Your code here
```

### 10. Backup Strategy

```bash
# MongoDB backup
mongodump --uri="mongodb+srv://username:password@cluster.mongodb.net/" --out=backup/

# Application backup
tar -czf app-backup-$(date +%Y%m%d).tar.gz /path/to/app

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mongodump --uri="$MONGODB_URI" --out="backup-$DATE"
tar -czf "app-backup-$DATE.tar.gz" /path/to/app
aws s3 cp "backup-$DATE" s3://your-backup-bucket/
```

### 11. Scaling

#### Horizontal Scaling

```yaml
# docker-compose.yml with multiple instances
version: '3.8'

services:
  mongodb-query-translator:
    build: .
    ports:
      - "8000-8002:8000"
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    deploy:
      replicas: 3
    restart: unless-stopped
```

#### Load Balancer

```nginx
upstream mongodb_query_translator {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://mongodb_query_translator;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Maintenance

### Regular Tasks

1. **Monitor logs** for errors and performance issues
2. **Update dependencies** monthly
3. **Backup database** daily
4. **Check API quotas** weekly
5. **Review security** quarterly

### Troubleshooting

#### Common Issues

1. **API quota exceeded**: Check OpenAI usage and upgrade plan
2. **Database connection issues**: Verify MongoDB URI and network access
3. **Memory issues**: Increase server RAM or optimize queries
4. **Slow responses**: Check database indexes and query optimization

#### Log Analysis

```bash
# Check application logs
tail -f logs/app.log

# Check error logs
grep "ERROR" logs/app.log

# Check performance
grep "execution_time" logs/app.log | sort -n
```

## Support

For deployment issues:
1. Check the logs in `logs/` directory
2. Verify environment variables
3. Test database connectivity
4. Review API documentation
5. Contact support team
