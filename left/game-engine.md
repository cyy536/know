tool layer:游戏开发编辑器，资产导入导出器

function layer:逻辑计算，渲染

resource layer:资产管理和垃圾回收

core layer:数学库，stl，内存管理

platform layer:通用图形库

game object用组合而非继承，每个组件都有tick函数，随时间更新

game object通过event交互

渲染：mesh，材质，纹理，shader code(由GPU执行处理mesh材质纹理)
