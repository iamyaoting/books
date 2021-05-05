using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text.RegularExpressions;

public class Export : MonoBehaviour
{
    //public Animator anim;
    public GameObject animGameObject;
    Transform[] jointTranforms;
    string line;
    StreamWriter bvhWriter;

    Dictionary<string, int> bvhJointIndexDic;
    Dictionary<int, int> bvh2unityJointMap;

    bool unityConsFlag = false;

    Vector3 posLast;

    // Start is called before the first frame update
    void Start()
    {
        bvhJointIndexDic = new Dictionary<string, int>();
        bvh2unityJointMap = new Dictionary<int, int>();
        // 读写bvh
        StreamReader bvhReader = new StreamReader("E:/Sissi_Personal/bvh-fbx/rabbit.bvh");
        bvhWriter = new StreamWriter("E:/Sissi_Personal/bvh-fbx/rabbit-1.bvh");

        posLast = new Vector3();

        int jointindex = 0;
        while ((line = bvhReader.ReadLine()) != null)
        {
            // construct jointname-index dic
            if (Regex.IsMatch(line, "ROOT"))
            {
                string jointname = line.Substring(line.Trim('\n').LastIndexOf(" ") + 1);
                bvhJointIndexDic.Add(jointname, jointindex);

                jointindex++;
            }

            if (Regex.IsMatch(line, "JOINT"))
            {
                string jointname = line.Substring(line.Trim('\n').LastIndexOf(" ") + 1);
                bvhJointIndexDic.Add(jointname, jointindex);

                jointindex++;
            }

            if (Regex.IsMatch(line, "MOTION"))
            {
                //Debug.Log(line);
                break;
            }
        }

        jointTranforms = animGameObject.GetComponentsInChildren<Transform>();
        // 2可能要修改
        for (int i = 2; i < jointTranforms.Length; i++)
        {
            bvh2unityJointMap.Add(bvhJointIndexDic[jointTranforms[i].name], i);
            //Debug.Log(jointTranforms[i].name);
        }
        bvhReader.Close();

        bvhReader = new StreamReader("E:/Sissi_Personal/bvh-fbx/rabbit.bvh");
        jointindex = 0;
        while ((line = bvhReader.ReadLine()) != null)
        {
            if (Regex.IsMatch(line, "OFFSET"))
            {
                string newline = "";
                newline += line.Substring(0, line.Trim('\n').LastIndexOf("OFFSET"));
                newline += "OFFSET " + posLast.x.ToString("f6") + " " + posLast.y.ToString("f6") + " " + posLast.z.ToString("f6");
                line = newline;
                Debug.Log(newline);
            }

            bvhWriter.WriteLine(line);
            bvhWriter.Flush();

            // construct jointname-index dic
            if (Regex.IsMatch(line, "ROOT")) { 
                //string jointname = line.Substring(line.Trim('\n').LastIndexOf(" ") + 1);
                //bvhJointIndexDic.Add(jointname, jointindex);

                posLast = jointTranforms[bvh2unityJointMap[jointindex]].position;
                posLast = posLast * 100;
                Debug.Log(posLast);

                jointindex++;
            }

            if (Regex.IsMatch(line, "JOINT")) {
                //string jointname = line.Substring(line.Trim('\n').LastIndexOf(" ") + 1);
                //bvhJointIndexDic.Add(jointname, jointindex);

                posLast = jointTranforms[bvh2unityJointMap[jointindex]].localPosition;
                posLast = posLast * 100;
                Debug.Log(posLast);

                jointindex++;
            }

            if (Regex.IsMatch(line, "MOTION"))
            {
                //Debug.Log(line);
                break;
            }
        }


        StreamReader bvhReader1 = new StreamReader("E:/Sissi_Personal/bvh-fbx/0005_Cartwheel001.bvh");
        int count = 0;
        while ((line = bvhReader1.ReadLine()) != null) {
            if (Regex.IsMatch(line, "Frame")) {
                bvhWriter.WriteLine(line);
                bvhWriter.Flush();
                count++;
                if (count == 2) {
                    break;
                }
            }
        }
        bvhReader.Close();
        bvhReader1.Close();
    }

    // Update is called once per frame
    void Update()
    {
        //jointTranforms = animGameObject.GetComponentsInChildren<Transform>();
        //if (!unityConsFlag) {
        //    // 2可能要修改
        //    for (int i = 2; i < jointTranforms.Length; i++) {
        //        bvh2unityJointMap.Add(bvhJointIndexDic[jointTranforms[i].name], i);
        //        //Debug.Log(jointTranforms[i].name);
        //    }
        //    unityConsFlag = true;
        //}
        string line = "";
        foreach (var item in bvhJointIndexDic)
        {
            //Debug.Log(jointTranforms[bvh2unityJointMap[item.Value]].name);
            if (Regex.IsMatch(jointTranforms[bvh2unityJointMap[item.Value]].name, "Hips"))
            {
                Vector3 pos = jointTranforms[bvh2unityJointMap[item.Value]].position;
                pos = pos * 100;
                Vector3 rot = jointTranforms[bvh2unityJointMap[item.Value]].localEulerAngles;
                line += pos.x.ToString("f6") + " " + pos.y.ToString("f6") + " " + pos.z.ToString("f6") + " ";
                line += rot.z.ToString("f6") + " " + rot.x.ToString("f6") + " " + rot.y.ToString("f6") + " ";
            } else {
                Vector3 pos = jointTranforms[bvh2unityJointMap[item.Value]].localPosition;
                pos = pos * 100;
                Vector3 rot = jointTranforms[bvh2unityJointMap[item.Value]].localEulerAngles;
                line += rot.z.ToString("f6") + " " + rot.x.ToString("f6") + " " + rot.y.ToString("f6") + " ";
            }
        }
        //Debug.Log(line);
        bvhWriter.WriteLine(line);
        bvhWriter.Flush();
    }
}
