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



        // // Parse 按钮
        // if (GUILayout.Button("Parse All Prefabs"))
        // {
        //     ParsePrefabsInAssets();
        // }
        // Debug.Log("procthorDatabasePath: " + procthorDatabasePath);

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

        // foreach (var assetGroupKey in jsonData.Keys)
        // {
        //     // 获取每个 asset group 中的 asset 列表
        //     // var assetList = jsonData[assetGroupKey] as List<object>;
        //     var assetList = jsonData[assetGroupKey] as List<Dictionary<string, object>>;

        //     // var assetList = jsonData[assetGroupKey] as JArray;

        //     if (assetList != null)
        //     {
        //         // 遍历每个 asset
        //         foreach (var assetObj in assetList)
        //         {
        //             var asset = assetObj as Dictionary<string, object>;

        //             if (asset != null && asset.ContainsKey("assetId"))
        //             {
        //                 // 获取 assetId
        //                 string assetId = asset["assetId"].ToString();
        //                 Debug.Log($"Asset ID: {assetId} in group: {assetGroupKey}");

                        
        //                 // 你可以在这里对 assetId 进行进一步处理
        //                 // 例如，加载对应的 prefab，或者其他操作
        //             }
        //         }
        //     }
        //     else
        //     {
        //         Debug.LogError($"Asset group {assetGroupKey} is not a valid list of assets.");
        //     }


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



        // // create a big dictionary to store all prefab details
        // Dictionary<string, Dictionary<string, Dictionary<string, object>>> allPrefabDetails = new Dictionary<string, Dictionary<string, Dictionary<string, object>>>();

        // foreach (string prefabGUID in allPrefabs)
        // {
        //     string prefabPath = AssetDatabase.GUIDToAssetPath(prefabGUID);
        //     GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);

        //     if (prefab != null)
        //     {
        //         // get the details of each prefab
        //         if (jsonData != null)
        //         {
        //             // do something with the loaded JSON data
        //             Debug.Log("Loaded JSON data:");
        //             foreach (var key in jsonData.Keys)
        //             {
        //                 Debug.Log($"Key: {key}");
        //             }
        //         }
        //         Dictionary<string, Dictionary<string, object>> prefabDetails = GetPrefabDetails(prefab);

        //         // TODO: get aabb box center here

                
        //         allPrefabDetails[prefab.name] = prefabDetails;
        //     }
        // }

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
        // 检查 simObj 是否为 null
        if (simObj != null)
        {
            // simObj 存在，可以安全地使用它
            // Debug.Log("SimObjPhysics component found.");
            
            // 进一步检查 AxisAlignedBoundingBox 是否为 null
            if (simObj.AxisAlignedBoundingBox != null)
            {
                boxCenter = simObj.AxisAlignedBoundingBox.center;

                // 使用 box 继续执行操作...
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



        foreach (MeshFilter mf in prefab.GetComponentsInChildren<MeshFilter>())
        {
            Transform objTransform = mf.transform;
            Mesh mesh = mf.sharedMesh;
            if (mf.sharedMesh == null)
            {
                Debug.LogWarning($"MeshFilter on object {mf.gameObject.name} has no mesh.");
                continue;
            }


            string assetPath = AssetDatabase.GetAssetPath(mesh);
            Transform originalTransform = null;
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
                        


                        // test
                        if (prefab.name == "Toilet_1")
                        {
                            Debug.Log($"Mesh: {mesh.name}, Position: {originalTransform.position}, Rotation: {originalTransform.rotation.eulerAngles}, Scale: {originalTransform.localScale}");

                        }

                        // test
                        if (prefab.name == "Plunger_3")
                        {
                            Debug.Log("Plunger here");
                            Debug.Log($"Position: {fbxPrefab.transform.position}, Rotation: {fbxPrefab.transform.rotation.eulerAngles}, Scale: {fbxPrefab.transform.localScale}");


                        }




                    }
                }
            }
        


            // Vector3 center = mesh.bounds.center;
            
            // pivot to center in world koordinates
            // Vector3 pivotToCenter = center;
            // Vector3 pivotToCenter = objTransform.TransformPoint(center);

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

                // {"pivotToCenter", new Dictionary<string, float> {
                //     { "x", pivotToCenter.x },
                //     { "y", pivotToCenter.y },
                //     { "z", pivotToCenter.z }
                // }},


                // {"boundingBoxTransform", new Dictionary<string, object> {
                //     { "Position", new Dictionary<string, float> {
                //         { "x", boxColliderPosition.x },
                //         { "y", boxColliderPosition.y },
                //         { "z", boxColliderPosition.z } }
                //     },
                //     { "Rotation", new Dictionary<string, float> {
                //         { "x", boxColliderRotation.x },
                //         { "y", boxColliderRotation.y },
                //         { "z", boxColliderRotation.z },
                //         { "w", boxColliderRotation.w } }
                //     },
                //     { "Scale", new Dictionary<string, float> {
                //         { "x", boxColliderScale.x },
                //         { "y", boxColliderScale.y },
                //         { "z", boxColliderScale.z } }
                //     },
                //     {"Center", new Dictionary<string, float> {
                //         { "x", boxColliderCenter.x },
                //         { "y", boxColliderCenter.y },
                //         { "z", boxColliderCenter.z }
                //     }},
                //     {"Size", new Dictionary<string, float> {
                //         { "x", boxColliderSize.x },
                //         { "y", boxColliderSize.y },
                //         { "z", boxColliderSize.z }
                //     }}
                // }},


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
            string meshKey = $"{mf.gameObject.name}{mesh.name}";
            prefabDetails[meshKey] = meshFilterDetails;
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
   
    // private static Transform FindMeshFilterInHierarchy(Transform parent, string meshName)
    // {
    //     foreach (Transform child in parent)
    //     {
    //         if (child.name == meshName)
    //         {
    //             return child;
    //         }
    //         Transform result = FindMeshFilterInHierarchy(child, meshName);
    //         if (result != null)
    //         {
    //             return result;
    //         }
    //     }
    //     return null;
    // }

    // 递归查找FBX中的MeshFilter，根据Mesh名称匹配
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




    // private Transform FindOriginalMeshTransform(GameObject fbxPrefab, string meshName)
    // {
    //     MeshFilter[] fbxMeshFilters = fbxPrefab.GetComponentsInChildren<MeshFilter>();

    //     foreach (MeshFilter fbxMeshFilter in fbxMeshFilters)
    //     {
    //         if (fbxMeshFilter.sharedMesh.name == meshName)
    //         {
                
    //             return fbxMeshFilter.transform;
    //         }
    //     }

    //     return null;
    // }

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

    // private Dictionary<string, object> LoadJsonFile(string path)
    // {
    //     try
    //     {
    //         if (File.Exists(path))
    //         {
    //             // 读取 JSON 文件内容
    //             string jsonContent = File.ReadAllText(path);

    //             // 解析 JSON 文件为字典并返回
    //             Dictionary<string, object> data = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonContent);
                
    //             // 在此处你可以进行数据处理
    //             Debug.Log("JSON file loaded successfully!");
    //             return data;
    //         }
    //         else
    //         {
    //             Debug.LogError("File not found: " + path);
    //             return null;
    //         }
    //     }
    //     catch (System.Exception e)
    //     {
    //         Debug.LogError("Failed to load JSON file: " + e.Message);
    //         return null;
    //     }
    // }

    // private Dictionary<string, object> LoadJsonFile(string path)
    // {
    //     try
    //     {
    //         if (File.Exists(path))
    //         {
    //             // 读取 JSON 文件内容
    //             string jsonContent = File.ReadAllText(path);
    //             Debug.Log("JSON Content: " + jsonContent); // 输出 JSON 内容

    //             // 尝试反序列化 JSON 文件为字典
    //             Dictionary<string, object> data = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonContent);

    //             if (data != null)
    //             {
    //                 Debug.Log("JSON file loaded and deserialized successfully!");
    //             }
    //             else
    //             {
    //                 Debug.LogError("Failed to deserialize JSON content. The structure might not match Dictionary<string, object>.");
    //             }

    //             return data;
    //         }
    //         else
    //         {
    //             Debug.LogError("File not found: " + path);
    //             return null;
    //         }
    //     }
    //     catch (System.Exception e)
    //     {
    //         Debug.LogError("Failed to load JSON file: " + e.Message);
    //         return null;
    //     }
    // }



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
        // 使用 AssetDatabase.FindAssets 来搜索文件名匹配的 Prefab
        string[] guids = AssetDatabase.FindAssets(assetId + " t:Prefab"); // 搜索类型为 Prefab 的文件

        if (guids.Length > 0)
        {
            // 取第一个匹配的 Prefab
            string path = AssetDatabase.GUIDToAssetPath(guids[0]);
            // Debug.Log($"Prefab found at: {path}");

            // 加载 Prefab
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);

            if (prefab != null)
            {
                // Debug.Log($"Prefab {assetId} loaded successfully from {path}");
                return prefab;
            }
        }

        Debug.LogError($"Prefab with name {assetId} not found in Assets folder.");
        return null;
    }

    private void SaveToJson(Dictionary<string, Dictionary<string, Dictionary<string, object>>> data, string fileName)
    {
        string json = JsonConvert.SerializeObject(data, Formatting.Indented);
        File.WriteAllText(fileName, json);
    }
}
