# Deploy to Railway (Free Hosting)

## Step 1: Create GitHub Repository

```bash
cd "/Users/justin/Desktop/Appliance Lookup Tool"

# Initialize git
git init
git add .
git commit -m "Initial commit - Appliance Lookup Tool"
```

Then create a new repository on GitHub:
1. Go to https://github.com/new
2. Name it: `appliance-lookup-tool`
3. Make it Private
4. Click "Create repository"

Then push your code:
```bash
git remote add origin https://github.com/YOUR_USERNAME/appliance-lookup-tool.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Railway

1. Go to https://railway.app
2. Click "Start a New Project"
3. Choose "Deploy from GitHub repo"
4. Select your `appliance-lookup-tool` repository
5. Railway will automatically detect it's a Python/Flask app

## Step 3: Add Environment Variables

In Railway dashboard:
1. Click on your project
2. Go to "Variables" tab
3. Click "New Variable"
4. Add: `GOOGLE_API_KEY` = `AIzaSyDLxzTDJaUz9ob8bNZJSaVDouKpAThMDUo`
5. Add: `FLASK_ENV` = `production`

## Step 4: Get Your URL

1. Go to "Settings" tab
2. Click "Generate Domain"
3. You'll get a URL like: `https://appliance-lookup-production.up.railway.app`

That's your permanent URL! Access it from anywhere.

## Alternative: Deploy to Render (Also Free)

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT app_launcher:app`
5. Add environment variable: `GOOGLE_API_KEY`
6. Click "Create Web Service"

You'll get a URL like: `https://appliance-lookup.onrender.com`

---

## Local Development

To run locally:
```bash
python3 app_launcher.py
```

Access at: http://127.0.0.1:5001
