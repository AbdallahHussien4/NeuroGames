using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class win : MonoBehaviour
{
    // Start is called before the first frame update
    public void Setup()
    {
        // make text visible
        gameObject.SetActive(true);
    }

    public void hide()
    {
        // make text invisible
        gameObject.SetActive(false);
    }
}
