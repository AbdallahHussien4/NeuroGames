using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using WebSocketSharp;
public class leftPunch : MonoBehaviour
{
    Animator left_animator;
    WebSocket ws;
    int action;
    public healthBar enemy;
    public healthBar player;
    public Animator fighter_animator;

    void Start()
    {
        //Get the Animator attached to the GameObject you are intending to animate.
        left_animator = gameObject.GetComponent<Animator>();

        // start a connection to the server
        ws = new WebSocket("ws://192.168.137.1:8080/");
        ws.Connect();

        // handling the received action
        ws.OnMessage += (sender, payload) => 
        {
            if(payload.Data == "2"){
                action = 2;
            }
        };

    }
    IEnumerator HitEffect()
    {
        yield return new WaitForSecondsRealtime(0.7f);
        
        // start the animation
        fighter_animator.SetTrigger("leftHitTrig");
        enemy.takeDamage(10);
    
    }
    void Update()
    {
        // check for the received action
        if ( action == 2) {
            if ( fighter_animator.GetCurrentAnimatorStateInfo(0).IsName("idle"))
            {
                if(enemy.getValue() > 0  && player.getValue() > 0)
                {
                    left_animator.Play("leftPunch", 0, 0.0f);
                    StartCoroutine(HitEffect());
                }
            }
            action = 0;
        }     
        
    }
}
