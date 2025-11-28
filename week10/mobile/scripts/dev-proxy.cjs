const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');
require('dotenv').config();

const target = process.env.TARGET_URL || 'http://localhost:8000';
const port = Number(process.env.PROXY_PORT || 3001);
const app = express();

app.use(cors({ origin: true }));
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,PATCH,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Tenant-ID, X-API-Key');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

const proxy = createProxyMiddleware({
  target,
  changeOrigin: true,
  ws: true,
  onProxyRes(proxyRes, req, res) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Expose-Headers', 'Content-Type');
  },
  onError(err, req, res) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    if (req.path === '/health') {
      res.status(200).json({ model: 'unavailable', kb_index: false, orders_db: false, metrics: {} });
      return;
    }
    if (req.path === '/models/list') {
      res.status(200).json({ current: 'unavailable', models: [] });
      return;
    }
    res.status(502).json({ code: -1, message: 'backend_unreachable', data: null });
  },
});

app.use('/', proxy);

app.listen(port, () => {
  console.log(`[dev-proxy] listening on http://localhost:${port} -> ${target}`);
});