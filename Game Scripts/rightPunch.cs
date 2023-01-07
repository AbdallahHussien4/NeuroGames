using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
public class rightPunch : MonoBehaviour
{
    Animator right_Animator;
    WebSocket ws;
    int action = 0;
    public Animator fighter_animator;
    public healthBar enemy;
    public healthBar player;

    void Start()
    {
        //Get the Animator attached to the GameObject you are intending to animate.
        right_Animator = gameObject.GetComponent<Animator>();

        // start connection to server
        ws = new WebSocket("ws://192.168.137.1:8080/");
        ws.Connect();

        // handle received action
        ws.OnMessage += (sender, payload) => {
            if(payload.Data == "1"){
                action = 1;
            }
        };
    }
    IEnumerator HitEffect()
    {
        yield return new WaitForSecondsRealtime(0.5f);
        
        // start animation
        fighter_animator.SetTrigger("rightHitTrig");
        enemy.takeDamage(10);
        
    }
    void Update()
    {
        
        // check the received action
        if (action == 1){

            if (fighter_animator.GetCurrentAnimatorStateInfo(0).IsName("idle"))
            {
                if(enemy.getValue() > 0  && player.getValue() > 0)
                {
                    right_Animator.Play("rightPunch", 0, 0.0f);
                    StartCoroutine(HitEffect());
                }
            }
            action = 0;
        }
    }
}
