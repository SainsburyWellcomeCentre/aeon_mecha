using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CombineModelWithFixedId
{
    public IObservable<PoseIdentityCollection> Process(IObservable<Tuple<PoseCollection, string>> source)
    {
        return source.Select(value =>
        {
            var poses = value.Item1;
            var id = value.Item2;
            var output = new PoseIdentityCollection(poses.Image, poses.Model);

            // Return empty output if no pose detected
            if (poses.Count == 0)
            {
                Console.WriteLine("No pose detected for this frame, returning empty output");
                return output;
            }
            // Return empty output if more than one pose detected
            if (poses.Count > 1)
            {
                Console.WriteLine("More than one pose detected for this frame. This code is meant for single animal sessions. Make sure to set the \"max instances\" parameter in SLEAP to 1. Returning empty output");
                return output;
            }
            var pose = poses[0];
            var poseIdentity = new PoseIdentity(pose.Image, pose.Model);
            poseIdentity.Centroid = pose.Centroid;
            foreach (var bodypart in pose)
            {
                poseIdentity.Add(bodypart);
            }
            poseIdentity.Identity = id;
            poseIdentity.Confidence = float.NaN;
            output.Add(poseIdentity);
            return output;
        });
    }
}
