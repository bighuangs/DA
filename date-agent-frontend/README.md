# 京析·数语 · 前端

个人电商经营智能问数与分析平台“京析·数语”的前端，面向内部电商运营人员，基于 Vue 3、Vite 和 ECharts，提供自然语言问数、销售与商品分析结果工作台、图表、仪表盘、元数据管理、权限管理和连续追问页面。目录名沿用 `date-agent-frontend`。

完整安装、后端依赖及安全说明见[项目 README](../README.md)。

## 开发与构建

建议使用 Node.js 22.12+。在本目录执行：

```bash
npm ci
npm run dev
```

默认地址为 `http://127.0.0.1:5173`，`/api` 代理到 `http://localhost:8000`。如需修改后端地址，在启动前设置进程环境变量 `VITE_API_PROXY_TARGET`。

```bash
npm run build
```

构建产物位于 `dist`。生产部署请使用静态服务器及 HTTPS/API 反向代理，不要直接暴露开发服务器。

## 测试

```bash
npx playwright install chromium
npm run test:e2e:mock
```

Mock 回归会启动自己的本地 Vite 服务，验证页面和交互合同。`npm run test:e2e:real` 需要已配置的真实后端及独立测试环境，可能访问或写入服务；运行前须检查对应配置，不应指向生产环境。

页面中存在某个功能入口不代表后端已启用该能力。评测页面尤其依赖可选的后端评测接口和额外资产；真实查询还需要完成工作空间、Schema、指标知识及检索索引初始化。
