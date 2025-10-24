# Production Deployment Guide

**Comprehensive guide for deploying Synthetic Consumer SSR API to production**

## Table of Contents

1. [Quick Start (Local Development)](#quick-start)
2. [Production Deployment Options](#production-deployment-options)
3. [Infrastructure Requirements](#infrastructure-requirements)
4. [Cost Analysis](#cost-analysis)
5. [Deployment Strategies](#deployment-strategies)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security Hardening](#security-hardening)
8. [Scaling Strategies](#scaling-strategies)
9. [Disaster Recovery](#disaster-recovery)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Local Development with Docker Compose

**Prerequisites**:
- Docker 20.10+ and Docker Compose 2.0+
- OpenAI API key (required)
- Google API key (optional, for Gemini)

**Steps**:

```bash
# 1. Clone repository
git clone https://github.com/your-repo/synthetic-consumer-ssr.git
cd synthetic_consumer_ssr

# 2. Configure environment
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-your-key-here
# GOOGLE_API_KEY=your-google-key

# 3. Start all services
docker-compose up -d

# 4. Verify deployment
curl http://localhost:8000/health

# 5. Access Swagger UI
open http://localhost:8000/docs

# 6. Monitor with Grafana
open http://localhost:3000  # admin/admin
```

**What's Included**:
- ✅ FastAPI application (port 8000)
- ✅ PostgreSQL database (port 5432)
- ✅ Redis cache (port 6379)
- ✅ Celery workers (background tasks)
- ✅ Flower UI (celery monitoring, port 5555)
- ✅ Prometheus metrics (port 9091)
- ✅ Grafana dashboards (port 3000)

---

## Production Deployment Options

### Option 1: Docker Compose (Recommended for Small-Medium Scale)

**Best for**: 50-500 surveys/day, single server deployment

```bash
# Use production docker-compose
docker-compose -f docker-compose.production.yml up -d
```

**Features**:
- 3 API replicas (load balanced by nginx)
- 5 Celery workers (parallel processing)
- Auto-restart on failure
- Resource limits (2 CPU, 4GB RAM per container)
- Structured logging

**Scaling**:
```bash
# Scale API instances
docker-compose -f docker-compose.production.yml up -d --scale api=5

# Scale Celery workers
docker-compose -f docker-compose.production.yml up -d --scale celery-worker=10
```

---

### Option 2: Kubernetes (Recommended for Enterprise Scale)

**Best for**: 500+ surveys/day, multi-region deployment, auto-scaling

**Step 1: Create Kubernetes manifests**

Create `k8s/` directory with the following files:

**`k8s/namespace.yaml`**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ssr-prod
```

**`k8s/configmap.yaml`**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ssr-config
  namespace: ssr-prod
data:
  ENV: "production"
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  ENABLE_DEMOGRAPHICS: "true"
  ENABLE_MULTI_SET_AVERAGING: "true"
  ENABLE_BIAS_DETECTION: "true"
  RATE_LIMIT_ENABLED: "true"
  METRICS_ENABLED: "true"
```

**`k8s/secrets.yaml`** (base64 encoded):
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ssr-secrets
  namespace: ssr-prod
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  google-api-key: <base64-encoded-key>
  database-url: <base64-encoded-url>
  redis-url: <base64-encoded-url>
  api-keys: <base64-encoded-keys>
```

**`k8s/deployment.yaml`**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ssr-api
  namespace: ssr-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ssr-api
  template:
    metadata:
      labels:
        app: ssr-api
    spec:
      containers:
      - name: api
        image: your-registry/ssr-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ssr-secrets
              key: openai-api-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ssr-secrets
              key: database-url
        envFrom:
        - configMapRef:
            name: ssr-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ssr-api-service
  namespace: ssr-prod
spec:
  selector:
    app: ssr-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Step 2: Deploy to Kubernetes**:

```bash
# Create namespace and secrets
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml

# Enable auto-scaling
kubectl autoscale deployment ssr-api \
  --namespace=ssr-prod \
  --cpu-percent=70 \
  --min=3 \
  --max=10

# Verify deployment
kubectl get pods -n ssr-prod
kubectl get svc -n ssr-prod
```

---

### Option 3: Cloud Platforms

#### AWS Deployment (ECS + Fargate)

**Best for**: Managed infrastructure, pay-per-use

```bash
# 1. Build and push image to ECR
aws ecr create-repository --repository-name ssr-api
docker build -t ssr-api .
docker tag ssr-api:latest ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/ssr-api:latest
aws ecr get-login-password | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/ssr-api:latest

# 2. Create ECS task definition (task-definition.json)
{
  "family": "ssr-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "ssr-api",
      "image": "${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/ssr-api:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "ENV", "value": "production"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "secrets": [
        {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "DATABASE_URL", "valueFrom": "arn:aws:secretsmanager:..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ssr-api",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "ssr"
        }
      }
    }
  ]
}

# 3. Create ECS service
aws ecs create-service \
  --cluster ssr-cluster \
  --service-name ssr-api \
  --task-definition ssr-api \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

**Additional AWS Services**:
- **RDS for PostgreSQL**: Managed database with automatic backups
- **ElastiCache for Redis**: Managed Redis with multi-AZ replication
- **Application Load Balancer**: SSL termination, health checks
- **CloudWatch**: Logs and metrics aggregation
- **Secrets Manager**: Secure API key storage

---

#### Google Cloud Platform (Cloud Run)

**Best for**: Serverless, auto-scaling to zero

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/${PROJECT_ID}/ssr-api

# 2. Deploy to Cloud Run
gcloud run deploy ssr-api \
  --image gcr.io/${PROJECT_ID}/ssr-api \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 100 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars ENV=production,LOG_LEVEL=INFO \
  --set-secrets OPENAI_API_KEY=openai-key:latest,DATABASE_URL=db-url:latest \
  --allow-unauthenticated
```

**Additional GCP Services**:
- **Cloud SQL for PostgreSQL**: Managed database
- **Memorystore for Redis**: Managed Redis
- **Cloud Load Balancing**: Global load distribution
- **Cloud Logging**: Centralized logs
- **Secret Manager**: Secure secrets

---

## Infrastructure Requirements

### Minimum Production Setup

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **API Server** | 2 vCPU, 4GB RAM | FastAPI application |
| **Database** | PostgreSQL 15, 2 vCPU, 4GB RAM, 100GB SSD | Survey data, results |
| **Cache** | Redis 7, 1 vCPU, 2GB RAM | Embeddings, rate limiting |
| **Workers** | 4x (1 vCPU, 2GB RAM each) | Background processing |
| **Monitoring** | 1 vCPU, 2GB RAM | Prometheus + Grafana |

**Total**: ~12 vCPU, 24GB RAM, 100GB storage

---

### Recommended Production Setup (500 surveys/day)

| Component | Specification | Quantity | Purpose |
|-----------|---------------|----------|---------|
| **API Servers** | 4 vCPU, 8GB RAM | 3 | Load balanced FastAPI |
| **Database** | PostgreSQL 15, 4 vCPU, 16GB RAM, 500GB SSD | 1 primary + 1 replica | HA database |
| **Cache** | Redis 7, 2 vCPU, 8GB RAM | 1 primary + 1 replica | HA cache |
| **Workers** | 2 vCPU, 4GB RAM | 10 | Parallel processing |
| **Load Balancer** | Managed service | 1 | SSL, routing |
| **Monitoring** | 2 vCPU, 4GB RAM | 1 | Prometheus + Grafana |

**Total**: ~50 vCPU, 136GB RAM, 500GB storage

---

## Cost Analysis

### LLM API Costs (Primary expense)

**GPT-4o Pricing** (as of 2024):
- Input: $2.50 per 1M tokens
- Output: $10.00 per 1M tokens

**Cost per Survey** (N=200 responses):
- Text elicitation: 200 requests × 150 tokens/request = 30,000 tokens
- Embeddings: 200 requests × 10 tokens/request = 2,000 tokens
- **Total per survey**: ~$0.40 - $0.60

**Monthly Costs** (500 surveys/day):
- 15,000 surveys/month × $0.50 = **$7,500/month**

### Infrastructure Costs

#### Docker Compose (VPS)

**DigitalOcean Droplet** (8 vCPU, 16GB RAM):
- Compute: $96/month
- Managed PostgreSQL: $60/month
- Managed Redis: $35/month
- **Total**: **$191/month**

#### AWS ECS + Fargate

**Compute**:
- 3 Fargate tasks (2 vCPU, 4GB each): $150/month
- 10 Celery workers (1 vCPU, 2GB each): $250/month

**Data**:
- RDS PostgreSQL (db.r5.large): $180/month
- ElastiCache Redis (cache.r5.large): $120/month

**Networking & Storage**:
- Application Load Balancer: $22/month
- Data transfer: $50/month
- EBS storage: $50/month

**Total**: **$822/month**

#### GCP Cloud Run

**Compute** (with auto-scaling):
- API instances: $200/month
- Cloud SQL: $180/month
- Memorystore Redis: $120/month
- Load balancing: $20/month

**Total**: **$520/month**

### Total Cost of Ownership (500 surveys/day)

| Component | Cost/Month | Percentage |
|-----------|------------|------------|
| **LLM API calls** | $7,500 | 90% |
| **Infrastructure** | $500-800 | 6-10% |
| **Monitoring/Logging** | $50-100 | 1% |
| **Total** | **~$8,000-8,500** | 100% |

**Cost Optimization Strategies**:
1. **LLM Caching**: Save 30-50% on repeated queries
2. **Batch Processing**: Group similar requests
3. **Temperature Optimization**: Lower temperature = fewer tokens
4. **Embedding Reuse**: Cache embeddings for 7+ days

---

## Deployment Strategies

### Blue-Green Deployment

**Zero-downtime deployments**:

```bash
# 1. Deploy green environment
docker-compose -f docker-compose.production.yml \
  -p ssr-green up -d

# 2. Run health checks
./scripts/health_check.sh http://green-api:8000

# 3. Switch load balancer to green
./scripts/switch_lb.sh green

# 4. Monitor for 10 minutes
./scripts/monitor.sh green 600

# 5. If stable, tear down blue
docker-compose -f docker-compose.production.yml \
  -p ssr-blue down
```

---

### Canary Deployment

**Gradual rollout to minimize risk**:

```bash
# 1. Deploy 10% of traffic to new version
kubectl set image deployment/ssr-api \
  api=your-registry/ssr-api:v2.0

kubectl scale deployment ssr-api-canary --replicas=1
kubectl scale deployment ssr-api-stable --replicas=9

# 2. Monitor error rates
./scripts/canary_metrics.sh

# 3. Gradually increase (25% → 50% → 100%)
kubectl scale deployment ssr-api-canary --replicas=3
kubectl scale deployment ssr-api-stable --replicas=7
```

---

## Monitoring & Observability

### Metrics to Track

**Application Metrics** (Prometheus):
```yaml
# Survey processing
- ssr_surveys_total{status}
- ssr_survey_duration_seconds
- ssr_responses_generated_total

# LLM API
- llm_requests_total{model,status}
- llm_request_duration_seconds
- llm_tokens_consumed_total{type}
- llm_cost_dollars_total

# System
- http_requests_total{method,path,status}
- http_request_duration_seconds
- celery_task_total{status}
```

**Grafana Dashboards**:
1. **Business Metrics**: Surveys/day, cost per survey, success rate
2. **Performance**: Response time, throughput, queue depth
3. **Infrastructure**: CPU, memory, disk, network
4. **Errors**: 4xx/5xx rates, LLM failures, task failures

### Alerting Rules

```yaml
# config/prometheus/alerts.yml
groups:
  - name: ssr_alerts
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"

    - alert: LLMAPIFailure
      expr: rate(llm_requests_total{status="error"}[5m]) > 0.1
      for: 2m
      labels:
        severity: critical

    - alert: HighCost
      expr: increase(llm_cost_dollars_total[1h]) > 500
      labels:
        severity: warning
      annotations:
        summary: "LLM costs exceed $500/hour"
```

---

## Security Hardening

### API Security Checklist

- ✅ **API Key Authentication**: Required for all endpoints
- ✅ **Rate Limiting**: 60 req/min, 1000 req/hour
- ✅ **CORS**: Whitelist allowed origins
- ✅ **HTTPS Only**: SSL/TLS termination at load balancer
- ✅ **Input Validation**: Pydantic schemas
- ✅ **SQL Injection Protection**: SQLAlchemy ORM
- ✅ **Secrets Management**: Environment variables, not code
- ✅ **Audit Logging**: All API calls logged

### Network Security

**Firewall Rules**:
```bash
# Allow only HTTPS from internet
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow PostgreSQL only from API servers
iptables -A INPUT -p tcp --dport 5432 -s ${API_SUBNET} -j ACCEPT

# Allow Redis only from API and workers
iptables -A INPUT -p tcp --dport 6379 -s ${INTERNAL_SUBNET} -j ACCEPT
```

---

## Scaling Strategies

### Horizontal Scaling

**Auto-scaling based on metrics**:

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ssr-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ssr-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

### Vertical Scaling

**Resource limits by workload**:

| Workload | CPU | Memory | Scaling Strategy |
|----------|-----|--------|------------------|
| API | 2-4 cores | 4-8 GB | Horizontal preferred |
| Celery Worker | 1-2 cores | 2-4 GB | Horizontal (10-50 workers) |
| Database | 4-8 cores | 16-32 GB | Vertical + read replicas |
| Redis | 2-4 cores | 8-16 GB | Vertical + cluster |

---

## Disaster Recovery

### Backup Strategy

**Automated Backups**:
```bash
# PostgreSQL daily backups (retain 30 days)
0 2 * * * pg_dump -h ${DB_HOST} -U ${DB_USER} ${DB_NAME} | \
  gzip > /backups/ssr_$(date +\%Y\%m\%d).sql.gz

# Upload to S3
aws s3 cp /backups/ssr_$(date +\%Y\%m\%d).sql.gz \
  s3://ssr-backups/daily/

# Redis snapshots every 6 hours
0 */6 * * * redis-cli BGSAVE
```

### Recovery Procedures

**Database Restoration**:
```bash
# 1. Stop API to prevent writes
docker-compose stop api celery-worker

# 2. Restore from backup
gunzip < ssr_backup.sql.gz | \
  psql -h ${DB_HOST} -U ${DB_USER} ${DB_NAME}

# 3. Verify data integrity
psql -h ${DB_HOST} -U ${DB_USER} -d ${DB_NAME} \
  -c "SELECT COUNT(*) FROM surveys;"

# 4. Restart services
docker-compose start api celery-worker
```

---

## Troubleshooting

### Common Issues

#### High LLM API Costs

**Symptoms**: Monthly costs >$10,000

**Solutions**:
1. Enable LLM caching: `ENABLE_LLM_CACHING=true`
2. Reduce cohort size for testing: `cohort_size=50`
3. Use lower temperature: `temperature=0.5`
4. Monitor with: `SELECT SUM(cost) FROM llm_requests WHERE created_at > NOW() - INTERVAL '1 day'`

#### Slow Survey Processing

**Symptoms**: Surveys taking >15 minutes

**Solutions**:
1. Scale Celery workers: `docker-compose up -d --scale celery-worker=20`
2. Increase concurrency: `CELERY_WORKER_CONCURRENCY=16`
3. Optimize database queries: Add indexes
4. Check LLM API latency: Monitor `llm_request_duration_seconds`

#### Out of Memory (OOM)

**Symptoms**: Containers restarting, OOM errors in logs

**Solutions**:
1. Increase container memory: `memory: 8Gi`
2. Reduce worker concurrency: `--concurrency=4`
3. Enable memory limits: `CELERY_MAX_TASKS_PER_CHILD=50`
4. Monitor: `docker stats`

#### Database Connection Pool Exhaustion

**Symptoms**: `FATAL: sorry, too many clients already`

**Solutions**:
1. Increase PostgreSQL max connections: `max_connections=200`
2. Use connection pooling: PgBouncer
3. Reduce API instances or worker concurrency
4. Monitor: `SELECT count(*) FROM pg_stat_activity`

---

## Production Checklist

### Pre-Deployment

- [ ] API keys configured and tested
- [ ] Database migrations executed
- [ ] Reference sets loaded
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Alerting rules tested
- [ ] Backup procedures automated
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Documentation updated

### Post-Deployment

- [ ] Health checks passing
- [ ] Metrics flowing to Prometheus
- [ ] Logs flowing to aggregator
- [ ] Alerts firing correctly
- [ ] Backup job ran successfully
- [ ] Performance meets SLAs
- [ ] Cost tracking enabled
- [ ] Team trained on runbooks

---

## Support & Resources

**Documentation**:
- [User Guide](docs/USER_GUIDE.md) - API usage and workflows
- [Technical Docs](docs/TECHNICAL.md) - Implementation details
- [API Reference](docs/API_REFERENCE.md) - Complete API spec

**Monitoring**:
- Grafana: http://your-domain:3000
- Prometheus: http://your-domain:9090
- Flower (Celery): http://your-domain:5555

**Logs**:
```bash
# View API logs
docker-compose logs -f api

# View worker logs
docker-compose logs -f celery-worker

# Search logs
docker-compose logs api | grep ERROR
```

---

**Questions? Open an issue on GitHub or contact the team.**
