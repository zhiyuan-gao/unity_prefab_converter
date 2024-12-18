using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
public class ParseAllPrefabs : EditorWindow
{
    private string saveJsonPath = "Assets/AllPrefabDetails.json";
    private string procthorDatabasePath = "/home/zgao/procthor/procthor/databases/asset-database.json";
    // private Dictionary<string, object> jsonData; 

    private JObject jsonData;

    [MenuItem("Tools/Parse All Prefabs in Assets")]
    public static void ShowWindow()
    {
        GetWindow<ParseAllPrefabs>("Parse All Prefabs in Assets");
    }




    private void OnGUI()
    {

        // Procthor Database Path 的布局
        GUILayout.Label("Procthor Database JSON Path:", EditorStyles.boldLabel);

        GUILayout.BeginHorizontal();
        GUILayout.Label("Path:", GUILayout.Width(40));

        // 地址栏输入框
        procthorDatabasePath = EditorGUILayout.TextField(procthorDatabasePath, GUILayout.MinWidth(150));

        GUILayout.Space(10); // 按钮与输入框之间的间隔

        if (GUILayout.Button("Browse", GUILayout.Width(80)))
        {
            string databasePath = EditorUtility.OpenFilePanel("Select JSON File", "", "json");
            if (!string.IsNullOrEmpty(databasePath))
            {
                procthorDatabasePath = databasePath;
                 // 加载并解析 JSON 文件
            }
        }
        GUILayout.EndHorizontal(); // 水平布局结束

        GUILayout.Space(30); // 添加空间


        GUILayout.Label("Save Path for Prefab-info:", EditorStyles.boldLabel);

        // 水平布局开始
        GUILayout.BeginHorizontal();
        GUILayout.Label("Path:", GUILayout.Width(40)); // 设置标签宽度

        // 地址栏输入框，保持合理的最小宽度
        saveJsonPath = EditorGUILayout.TextField(saveJsonPath, GUILayout.MinWidth(150));

        // 浏览按钮
        if (GUILayout.Button("Browse", GUILayout.Width(80)))
        {
            string savePath = EditorUtility.SaveFilePanel("Save JSON File", "Assets", "AllPrefabDetails", "json");
            if (!string.IsNullOrEmpty(savePath))
            {
                if (savePath.StartsWith(Application.dataPath))
                {
                    saveJsonPath = "Assets" + savePath.Substring(Application.dataPath.Length);
                }
                else
                {
                    saveJsonPath = savePath;
                }
            }
        }
        GUILayout.EndHorizontal(); // 水平布局结束

        GUILayout.Space(30); // 添加空间


        if (GUILayout.Button("Parse All Prefabs"))
        {   
            jsonData = LoadJsonFile(procthorDatabasePath); 
            if (jsonData != null)
            {
                ParsePrefabsInAssets();
            }
            else
            {
                Debug.LogError("jsonData is null. Please load the JSON file first.");
            }
        }

    }

    private void ParsePrefabsInAssets()
    {
        // find all prefabs in the Assets folder
        // string[] allPrefabs = AssetDatabase.FindAssets("t:Prefab", new[] { "Assets" });
        Debug.Log("start");
        if (jsonData == null)
        {
            Debug.LogError("jsonData is null. Please ensure the JSON file is loaded correctly.");
            return; 
        }
        Dictionary<string, Dictionary<string, Dictionary<string, object>>> allPrefabDetails = new Dictionary<string, Dictionary<string, Dictionary<string, object>>>();


        foreach (var assetGroupKey in jsonData.Properties())
        {
            // 获取每个 asset group 中的 asset 列表
            JArray assetList = jsonData[assetGroupKey.Name] as JArray;

            if (assetList != null)
            {
                // 遍历每个 asset
                foreach (JToken asset in assetList)
                {
                    
                    var assetDict = asset.ToObject<Dictionary<string, object>>();

                    if (assetDict != null && assetDict.ContainsKey("assetId"))
                    {
                        // 获取 assetId
                        string assetId = assetDict["assetId"].ToString();
                        // Debug.Log($"Asset ID: {assetId} in group: {assetGroupKey}");
                        GameObject prefab = LoadPrefabByName(assetId);

                        // GameObject spawned = Instantiate(prefab);
                        if (prefab != null)
                        {

                            Dictionary<string, Dictionary<string, object>> prefabDetails = GetPrefabDetails(prefab);

                            // TODO: get aabb box center here

                            allPrefabDetails[prefab.name] = prefabDetails;
                        }
                    }
                }
            }
            else
            {
                Debug.LogError($"Asset group {assetGroupKey} is not a valid list of assets.");
            }





        }


        // // save the dictionary to a JSON file
        SaveToJson(allPrefabDetails, saveJsonPath);
        Debug.Log($"All Prefab details saved to {saveJsonPath}");
    }


    private Dictionary<string, Dictionary<string, object>> GetPrefabDetails(GameObject prefab)
    {
        Transform rootTransform = prefab.transform;
        Dictionary<string, Dictionary<string, object>> prefabDetails = new Dictionary<string, Dictionary<string, object>>();

        var simObj = prefab.GetComponent<SimObjPhysics>();
        Vector3? boxCenter = null;
        // var bBox = prefab.GetComponent<boundingBox>();
        // 检查 simObj 是否为 null
        if (simObj != null)
        {

            // 进一步检查 AxisAlignedBoundingBox 是否为 null
            if (simObj.AxisAlignedBoundingBox != null)
            {
                // this boxCenter may be not correct
                boxCenter = simObj.AxisAlignedBoundingBox.center;

                Transform boundingBoxTransform = prefab.transform.Find("BoundingBox");
                if (boundingBoxTransform != null)
                {
                    // 获取 GameObject
                    GameObject boundingBoxObject = boundingBoxTransform.gameObject;

                    // 获取 BoxCollider 组件
                    BoxCollider boxCollider = boundingBoxObject.GetComponent<BoxCollider>();

                    if (boxCollider != null)
                    {
                        boxCenter = boxCollider.center;
                    }

                }


            }
            else
            {
                Debug.LogError("AxisAlignedBoundingBox is null for this simObj.");
            }
        }
        else
        {
            
            Debug.LogError("SimObjPhysics component is missing on the prefab.");
        }


        int meshCount = 0;
        foreach (MeshFilter mf in prefab.GetComponentsInChildren<MeshFilter>())
        {
            Transform objTransform = mf.transform;
            Mesh mesh = mf.sharedMesh;
            string meshPath = AssetDatabase.GetAssetPath(mf.sharedMesh);
            if (mf.sharedMesh == null)
            {
                Debug.LogWarning($"MeshFilter on object {mf.gameObject.name} has no mesh.");
                continue;
            }

            string assetPath = AssetDatabase.GetAssetPath(mesh);
            Transform originalTransform = null;
            float scaleFactor = 0;
            if (!string.IsNullOrEmpty(assetPath))
            {
                // Debug.Log("Mesh: " + mesh.name + " is from asset: " + assetPath);

                if (assetPath.EndsWith(".fbx", System.StringComparison.OrdinalIgnoreCase))
                {
                    
                    GameObject fbxPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);

                    if (fbxPrefab != null)
                    {
                        // originalTransform = FindOriginalMeshTransform(fbxPrefab, mesh.name);
                        originalTransform = FindMeshFilterInHierarchy(fbxPrefab.transform, mesh.name);
                        if (originalTransform == null)
                        {
                            originalTransform = fbxPrefab.transform;
                        }

                        ModelImporter modelImporter = AssetImporter.GetAtPath(meshPath) as ModelImporter;
                        if (modelImporter != null)
                        {
                            scaleFactor = modelImporter.globalScale;
                        }

                    }

                }
            }
        


            MeshRenderer meshRenderer = mf.GetComponent<MeshRenderer>();
            string hierarchyPath = GetHierarchyPath(mf.transform);
            // calculate the transform relative to the root
            Vector3 localToRootPosition = rootTransform.InverseTransformPoint(objTransform.position);
            Quaternion localToRootRotation = Quaternion.Inverse(rootTransform.rotation) * objTransform.rotation;
            Vector3 localToRootScale = new Vector3(
                objTransform.lossyScale.x / rootTransform.lossyScale.x,
                objTransform.lossyScale.y / rootTransform.lossyScale.y,
                objTransform.lossyScale.z / rootTransform.lossyScale.z
            );

            Dictionary<string, object> meshFilterDetails = new Dictionary<string, object>
            {
                { "HierarchyPath", hierarchyPath },

                { "Transform", new Dictionary<string, object> {
                    { "Position", new Dictionary<string, float> {
                        { "x", localToRootPosition.x },
                        { "y", localToRootPosition.y },
                        { "z", localToRootPosition.z } }
                    },
                    { "Rotation", new Dictionary<string, float> {
                        { "x", localToRootRotation.x },
                        { "y", localToRootRotation.y },
                        { "z", localToRootRotation.z },
                        { "w", localToRootRotation.w } }
                    },
                    { "Scale", new Dictionary<string, float> {
                        { "x", localToRootScale.x },
                        { "y", localToRootScale.y },
                        { "z", localToRootScale.z } }
                    }
                }},



                { "MeshName", mesh != null ? mesh.name : "None" },
                { "MeshPath", mesh != null ? AssetDatabase.GetAssetPath(mesh) : "None" }
            };



            if (originalTransform != null)
            {

                meshFilterDetails.Add("OriginalTransform", new Dictionary<string, object> {
                    { "Position", new Dictionary<string, float> {
                        { "x", originalTransform.position.x },
                        { "y", originalTransform.position.y },
                        { "z", originalTransform.position.z }
                    }},
                    { "Rotation", new Dictionary<string, float> {
                        { "x", originalTransform.rotation.x },
                        { "y", originalTransform.rotation.y },
                        { "z", originalTransform.rotation.z },
                        { "w", originalTransform.rotation.w }
                    }},
                    { "Scale", new Dictionary<string, float> {
                        { "x", originalTransform.lossyScale.x },
                        { "y", originalTransform.lossyScale.y },
                        { "z", originalTransform.lossyScale.z }
                    }}
                });

            }

            if (scaleFactor != 0)
            {
                meshFilterDetails.Add("ScaleFactor", scaleFactor);
            }

            // else
            // {
            //     Debug.LogWarning("OriginalTransform is null, not adding it to the dictionary.");
            // }

        
            if (boxCenter.HasValue)
            {
                meshFilterDetails.Add("BoxCenter", new Dictionary<string, object>{
                    { "x", boxCenter.Value.x },
                    { "y", boxCenter.Value.y },
                    { "z", boxCenter.Value.z }
                });   
            }


            if (meshRenderer != null)
            {
                Material[] materials = meshRenderer.sharedMaterials;
                List<Dictionary<string, string>> materialDetails = new List<Dictionary<string, string>>();

                foreach (Material mat in materials)
                {
                    if (mat != null)
                    {
                        materialDetails.Add(new Dictionary<string, string>
                        {
                            { "MaterialName", mat.name },
                            { "MaterialPath", AssetDatabase.GetAssetPath(mat) }
                        });
                    }
                    else
                    {
                        materialDetails.Add(new Dictionary<string, string>
                        {
                            { "MaterialName", "None" },
                            { "MaterialPath", "None" }
                        });
                    }
                }
                meshFilterDetails["Materials"] = materialDetails;
            }
            else
            {
                meshFilterDetails["Materials"] = new List<Dictionary<string, string>> { new Dictionary<string, string>
                {
                    { "MaterialName", "None" },
                    { "MaterialPath", "None" }
                }};
            }

        //     // add to the prefabDetails dictionary with the child object's name as the key
            string meshKey = $"{mf.gameObject.name}_{mesh.name}_{meshCount}";

            prefabDetails[meshKey] = meshFilterDetails;
            meshCount++;
        }


        return prefabDetails;
        // return null;
    }

    private string GetHierarchyPath(Transform transform)
    {
        string path = transform.name;
        while (transform.parent != null && transform.parent != transform.root)
        {
            transform = transform.parent;
            path = transform.name + "/" + path;
        }
        return path;
    }


    // Find the MeshFilter in the FBX hierarchy based on the mesh name
    private static Transform FindMeshFilterInHierarchy(Transform parent, string meshName)
    {
        // 遍历所有子节点
        foreach (Transform child in parent)
        {
            // 检查是否有MeshFilter组件
            MeshFilter meshFilter = child.GetComponent<MeshFilter>();
            if (meshFilter != null && meshFilter.sharedMesh != null && meshFilter.sharedMesh.name == meshName)
            {
                return meshFilter.transform;
            }
            // 递归检查子节点
            Transform result = FindMeshFilterInHierarchy(child, meshName);
            if (result != null)
            {
                return result;
            }

        }
        return null;
    }



    private Matrix4x4 GetGlobalTransformRelativeToRoot(Transform meshTransform)
    {
        Transform currentTransform = meshTransform;
        Matrix4x4 globalMatrix = currentTransform.localToWorldMatrix; 

        while (currentTransform.parent != null)
        {
            currentTransform = currentTransform.parent;
            globalMatrix = currentTransform.localToWorldMatrix * globalMatrix;
        }

        return globalMatrix; 
    }



    private JObject LoadJsonFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                // 读取 JSON 文件内容
                string jsonContent = File.ReadAllText(path);

                // 使用 JObject 解析 JSON 文件
                JObject jsonData = JObject.Parse(jsonContent);

                if (jsonData != null)
                {
                    // Debug.Log("JSON file loaded and parsed successfully!");
                }
                else
                {
                    Debug.LogError("Failed to parse JSON content.");
                }

                return jsonData;
            }
            else
            {
                Debug.LogError("File not found: " + path);
                return null;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to load JSON file: " + e.Message);
            return null;
        }
    }


    private GameObject LoadPrefabByName(string assetId)
    {
        // 使用 AssetDatabase.FindAssets 搜索类型为 Prefab 的文件
        string[] guids = AssetDatabase.FindAssets($"\"{assetId}\" t:Prefab"); // 精确搜索

        foreach (string guid in guids)
        {
            // 将 GUID 转换为路径
            string path = AssetDatabase.GUIDToAssetPath(guid);
            string fileName = System.IO.Path.GetFileNameWithoutExtension(path); // 获取文件名（不包括扩展名）

            // 检查文件名是否与 assetId 完全匹配
            if (fileName == assetId)
            {
                // 加载 Prefab
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);

                if (prefab != null)
                {
                    // Debug.Log($"Prefab {assetId} loaded successfully from {path}");
                    return prefab; // 返回第一个完全匹配的 Prefab
                }
            }
        }

        Debug.LogError($"Prefab with exact name {assetId} not found in Assets folder.");
        return null; // 如果找不到完全匹配的 Prefab，返回 null
    }



    private void SaveToJson(Dictionary<string, Dictionary<string, Dictionary<string, object>>> data, string fileName)
    {
        string json = JsonConvert.SerializeObject(data, Formatting.Indented);
        File.WriteAllText(fileName, json);
    }
}
